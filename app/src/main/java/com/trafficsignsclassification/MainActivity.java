package com.trafficsignsclassification;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.Settings;
import android.text.Spannable;
import android.text.SpannableString;
import android.text.style.ForegroundColorSpan;
import android.util.Log;
import android.util.Size;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final int CAMERA_REQUEST_CODE = 101;
    private static final String TAG = "MainActivity";
    private static final float THRESHOLD = 0.7f;

    // ===================================================================
    // BARU: Class kecil untuk menyimpan data rambu secara terstruktur
    // ===================================================================
    public static class SignInfo {
        String nameInIndonesian;
        String explanationInIndonesian;
    }
    // ===================================================================

    // Komponen UI
    private PreviewView previewView;
    private TextView classTextView, signExplanationTextView;
    private ImageButton flashToggleButton;
    private LinearLayout permissionLayout;

    // TFLite & Data
    private Interpreter tflite;
    private List<String> labels;
    // ✅ PERUBAHAN: Tipe data Map diubah untuk menyimpan objek SignInfo
    private Map<String, SignInfo> signDataMap;

    // CameraX
    private ExecutorService cameraExecutor;
    private Camera camera;
    private boolean isFlashOn = false;

    // Variabel untuk membatasi pembaruan UI
    private final Handler uiUpdateHandler = new Handler(Looper.getMainLooper());
    private Runnable uiUpdateRunnable;
    private volatile String latestDetectedSign = "N/A";
    private final long UI_UPDATE_DELAY = 1000L;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView = findViewById(R.id.previewView);
        classTextView = findViewById(R.id.classTextView);
        TextView probabilityTextView = findViewById(R.id.probabilityTextView);
        signExplanationTextView = findViewById(R.id.signExplanationTextView);
        flashToggleButton = findViewById(R.id.flashToggleButton);
        permissionLayout = findViewById(R.id.permissionLayout);
        Button openSettingsButton = findViewById(R.id.openSettingsButton);

        probabilityTextView.setVisibility(View.GONE);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getWindow().setStatusBarColor(ContextCompat.getColor(this, R.color.customBlack));

        flashToggleButton.setOnClickListener(v -> toggleFlashlight());
        openSettingsButton.setOnClickListener(v -> openAppSettings());

        loadLabels();
        loadSignData(); // ✅ PERUBAHAN: Memanggil method baru
        loadTFLiteModel();

        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "Gagal memuat OpenCV");
        }

        cameraExecutor = Executors.newSingleThreadExecutor();

        uiUpdateRunnable = () -> {
            updateUI();
            uiUpdateHandler.postDelayed(uiUpdateRunnable, UI_UPDATE_DELAY);
        };

        checkCameraPermission();
    }

    private void checkCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            showCameraViews();
            startCamera();
        } else {
            hideCameraViews();
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, CAMERA_REQUEST_CODE);
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());
                ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                        .setTargetResolution(new Size(640, 480))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();
                imageAnalysis.setAnalyzer(cameraExecutor, this::processImage);
                CameraSelector cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;
                cameraProvider.unbindAll();
                camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);
                updateFlashButtonState();
            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "Gagal mengikat use case kamera", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }


    @SuppressLint("UnsafeOptInUsageError")
    private void processImage(ImageProxy imageProxy) {
        if (imageProxy == null || imageProxy.getImage() == null) { return; }

        Bitmap bitmap = toBitmap(imageProxy);
        if (bitmap == null) { imageProxy.close(); return; }

        Matrix matrix = new Matrix();
        matrix.postRotate(imageProxy.getImageInfo().getRotationDegrees());
        Bitmap rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);

        int frameWidth = rotatedBitmap.getWidth();
        int frameHeight = rotatedBitmap.getHeight();
        int cropSize = Math.min(frameWidth, frameHeight) / 2;
        int x = (frameWidth - cropSize) / 2;
        int y = (frameHeight - cropSize) / 2;
        Bitmap croppedBitmap = Bitmap.createBitmap(rotatedBitmap, x, y, cropSize, cropSize);

        Mat mat = new Mat();
        Utils.bitmapToMat(croppedBitmap, mat);
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2GRAY);
        Imgproc.equalizeHist(mat, mat);
        Mat resizedMat = new Mat();
        Imgproc.resize(mat, resizedMat, new org.opencv.core.Size(32, 32));
        resizedMat.convertTo(resizedMat, CvType.CV_32F, 1.0 / 255);
        float[][][][] input = new float[1][32][32][1];
        for (int i = 0; i < 32; i++) {
            for (int j = 0; j < 32; j++) {
                input[0][i][j][0] = (float) resizedMat.get(i, j)[0];
            }
        }
        float[][] output = new float[1][labels.size()];
        tflite.run(input, output);

        int classIndex = getMaxIndex(output[0]);
        float probability = output[0][classIndex];

        if (probability > THRESHOLD && classIndex != -1) {
            latestDetectedSign = labels.get(classIndex);
        } else {
            latestDetectedSign = "N/A";
        }

        imageProxy.close();
    }

    // ✅ PERUBAHAN: Logika update UI untuk mengambil nama dan penjelasan dari SignInfo
    private void updateUI() {
        int customGreen = ContextCompat.getColor(this, R.color.deep_green);
        if (!latestDetectedSign.equals("N/A")) {
            SignInfo currentSign = signDataMap.get(latestDetectedSign);

            if (currentSign != null) {
                // Tampilkan NAMA dalam Bahasa Indonesia
                SpannableString classText = new SpannableString("Rambu: " + currentSign.nameInIndonesian);
                classText.setSpan(new ForegroundColorSpan(customGreen), 7, classText.length(), Spannable.SPAN_EXCLUSIVE_EXCLUSIVE);
                classTextView.setText(classText);

                // Tampilkan PENJELASAN dalam Bahasa Indonesia
                signExplanationTextView.setText(currentSign.explanationInIndonesian);
            }
        } else {
            // Tampilkan status "N/A"
            SpannableString classText = new SpannableString("Rambu: Tidak Dikenali");
            classText.setSpan(new ForegroundColorSpan(Color.RED), 7, classText.length(), Spannable.SPAN_EXCLUSIVE_EXCLUSIVE);
            classTextView.setText(classText);

            signExplanationTextView.setText("Arahkan kamera ke rambu lalu lintas untuk memulai.");
        }
    }

    private void toggleFlashlight() {
        if (camera != null && camera.getCameraInfo().hasFlashUnit()) {
            isFlashOn = !isFlashOn;
            camera.getCameraControl().enableTorch(isFlashOn);
            updateFlashButtonState();
        } else {
            Toast.makeText(this, "Flash tidak tersedia.", Toast.LENGTH_SHORT).show();
        }
    }

    private void updateFlashButtonState() {
        if (camera != null && camera.getCameraInfo().hasFlashUnit()) {
            flashToggleButton.setVisibility(View.VISIBLE);
            Integer torchState = camera.getCameraInfo().getTorchState().getValue();
            isFlashOn = torchState != null && torchState == 1;
            flashToggleButton.setImageResource(isFlashOn ? R.drawable.outline_flash_off_24 : R.drawable.outline_flash_on_24);
        } else {
            flashToggleButton.setVisibility(View.GONE);
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        uiUpdateHandler.post(uiUpdateRunnable);
        new Handler().postDelayed(this::checkCameraPermission, 500);
    }

    @Override
    protected void onPause() {
        super.onPause();
        uiUpdateHandler.removeCallbacks(uiUpdateRunnable);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraExecutor.shutdown();
        if (tflite != null) {
            tflite.close();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_REQUEST_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                showCameraViews();
                startCamera();
            } else {
                hideCameraViews();
                Toast.makeText(this, "Izin kamera diperlukan untuk menggunakan aplikasi ini.", Toast.LENGTH_LONG).show();
            }
        }
    }

    // ===================================================================
    // PERUBAHAN BESAR: Method ini sekarang memuat NAMA dan PENJELASAN ke dalam objek SignInfo
    // ===================================================================
    private void loadSignData() {
        signDataMap = new HashMap<>();
        SignInfo sign;

        // --- Penjelasan Rambu Versi Ramah Anak (LENGKAP 43 KELAS) ---

        // Rambu Batas Kecepatan
        sign = new SignInfo();
        sign.nameInIndonesian = "Batas Kecepatan 20 km/j";
        sign.explanationInIndonesian = "Artinya, mobil di sini jalannya harus pelan-pelan, seperti kura-kura! Tidak boleh lebih cepat dari angka 20.";
        signDataMap.put("Speed limit (20km/h)", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Batas Kecepatan 30 km/j";
        sign.explanationInIndonesian = "Di jalan ini, mobil boleh sedikit lebih cepat, tapi tetap pelan ya, maksimal di angka 30.";
        signDataMap.put("Speed limit (30km/h)", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Batas Kecepatan 50 km/j";
        sign.explanationInIndonesian = "Ini batas kecepatan di kota. Mobil harus melaju santai, tidak boleh melebihi angka 50.";
        signDataMap.put("Speed limit (50km/h)", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Batas Kecepatan 60 km/j";
        sign.explanationInIndonesian = "Kecepatan mobil di sini maksimal 60 ya. Tidak terlalu cepat, tidak terlalu pelan.";
        signDataMap.put("Speed limit (60km/h)", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Batas Kecepatan 70 km/j";
        sign.explanationInIndonesian = "Mobil boleh sedikit ngebut, tapi jangan sampai melewati angka 70 ya!";
        signDataMap.put("Speed limit (70km/h)", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Batas Kecepatan 80 km/j";
        sign.explanationInIndonesian = "Ini adalah batas kecepatan di jalan yang lebih besar. Maksimal di angka 80.";
        signDataMap.put("Speed limit (80km/h)", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Akhir Batas Kecepatan 80 km/j";
        sign.explanationInIndonesian = "Hore! Batas kecepatan 80 sudah selesai. Sekarang mobil boleh jalan dengan kecepatan normal lagi.";
        signDataMap.put("End of speed limit (80km/h)", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Batas Kecepatan 100 km/j";
        sign.explanationInIndonesian = "Wiuussh! Ini jalan tol, mobil boleh melaju cepat sampai angka 100.";
        signDataMap.put("Speed limit (100km/h)", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Batas Kecepatan 120 km/j";
        sign.explanationInIndonesian = "Super cepat! Ini kecepatan maksimal di jalan tol, sampai angka 120. Seru!";
        signDataMap.put("Speed limit (120km/h)", sign);

        // Rambu Larangan
        sign = new SignInfo();
        sign.nameInIndonesian = "Dilarang Menyalip";
        sign.explanationInIndonesian = "Hore, tidak boleh balapan! Mobil harus tetap di belakang teman di depannya, tidak boleh menyalip.";
        signDataMap.put("No passing", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Truk Dilarang Menyalip";
        sign.explanationInIndonesian = "Truk besar yang berat tidak boleh menyalip mobil lain di sini.";
        signDataMap.put("No passing for vehicles over 3.5 tons", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Dahulukan dari Arah Berlawanan";
        sign.explanationInIndonesian = "Di persimpangan depan, kamu dapat giliran jalan duluan! Tapi tetap hati-hati ya.";
        signDataMap.put("Right-of-way at the next intersection", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Jalan Utama";
        sign.explanationInIndonesian = "Kamu ada di jalan raja! Mobil dari jalan kecil harus menunggu kamu lewat dulu. Hebat, kan?";
        signDataMap.put("Priority road", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Wajib Beri Jalan";
        sign.explanationInIndonesian = "Lihat! Ada teman mau lewat. Kita kasih jalan dulu ya, supaya semua aman dan tidak tabrakan.";
        signDataMap.put("Yield", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Berhenti";
        sign.explanationInIndonesian = "Berhenti! Seperti main patung, semua mobil harus berhenti total. Lihat kanan dan kiri, kalau sudah aman baru boleh jalan lagi.";
        signDataMap.put("Stop", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Kendaraan Dilarang Masuk";
        sign.explanationInIndonesian = "Ups, semua mobil, motor, dan truk tidak boleh lewat jalan ini.";
        signDataMap.put("No vehicles", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Truk Dilarang Masuk";
        sign.explanationInIndonesian = "Truk besar dilarang masuk ke jalan ini, mungkin karena jalannya sempit.";
        signDataMap.put("Vehicles over 3.5 tons prohibited", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Dilarang Masuk";
        sign.explanationInIndonesian = "Eits, tidak boleh masuk! Ini jalan terlarang. Kalau masuk nanti bisa salah jalan atau bertemu mobil dari depan.";
        signDataMap.put("No entry", sign);

        // Rambu Peringatan
        sign = new SignInfo();
        sign.nameInIndonesian = "Peringatan Hati-Hati";
        sign.explanationInIndonesian = "Awas, hati-hati! Ada sesuatu di depan. Kurangi kecepatan dan lihat baik-baik ya.";
        signDataMap.put("General caution", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Tikungan Tajam ke Kiri";
        sign.explanationInIndonesian = "Ada tikungan tajam ke kiri di depan. Pegangan yang erat!";
        signDataMap.put("Dangerous curve to the left", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Tikungan Tajam ke Kanan";
        sign.explanationInIndonesian = "Siap-siap, di depan ada tikungan tajam ke kanan!";
        signDataMap.put("Dangerous curve to the right", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Tikungan Ganda";
        sign.explanationInIndonesian = "Wow, ada dua tikungan berturut-turut! Belok pertama, lalu belok lagi. Seru!";
        signDataMap.put("Double curve", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Jalan Bergelombang";
        sign.explanationInIndonesian = "Jalannya tidak rata! Siap-siap, mobil akan sedikit bergoyang seperti naik kuda. Pegangan ya!";
        signDataMap.put("Bumpy road", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Jalan Licin";
        sign.explanationInIndonesian = "Hati-hati, jalannya licin! Apalagi kalau hujan, mobil harus jalan pelan-pelan agar tidak tergelincir.";
        signDataMap.put("Slippery road", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Jalan Menyempit di Kanan";
        sign.explanationInIndonesian = "Jalannya menyempit di sebelah kanan. Mobil-mobil harus sedikit merapat.";
        signDataMap.put("Road narrows on the right", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Ada Perbaikan Jalan";
        sign.explanationInIndonesian = "Ada kakak-kakak pekerja sedang memperbaiki jalan. Kita jalan pelan-pelan ya agar tidak mengganggu mereka.";
        signDataMap.put("Road work", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Lampu Lalu Lintas";
        sign.explanationInIndonesian = "Lihat, ada lampu lalu lintas di depan! Perhatikan warnanya ya, merah berhenti, kuning hati-hati, hijau jalan.";
        signDataMap.put("Traffic signals", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Hati-Hati Pejalan Kaki";
        sign.explanationInIndonesian = "Ini tempat orang menyeberang jalan. Mobil harus berhenti dan mempersilakan mereka lewat dulu.";
        signDataMap.put("Pedestrians", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Hati-Hati Anak-Anak";
        sign.explanationInIndonesian = "Hati-hati! Di sini banyak anak-anak menyeberang, jadi semua mobil harus super pelan.";
        signDataMap.put("Children crossing", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Hati-Hati Sepeda";
        sign.explanationInIndonesian = "Banyak sepeda akan lewat di sini. Kasih jalan untuk para pengendara sepeda ya.";
        signDataMap.put("Bicycles crossing", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Waspada Jalan Bersalju/Es";
        sign.explanationInIndonesian = "Brrr, dingin! Jalannya bisa ada es atau salju dan jadi sangat licin. Hati-hati!";
        signDataMap.put("Beware of ice/snow", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Hati-Hati Hewan Liar";
        sign.explanationInIndonesian = "Awas! Mungkin ada hewan seperti rusa atau kancil yang mau menyeberang jalan. Pelan-pelan ya.";
        signDataMap.put("Wild animals crossing", sign);

        // Rambu Perintah dan Akhir Batas
        sign = new SignInfo();
        sign.nameInIndonesian = "Akhir Semua Batasan";
        sign.explanationInIndonesian = "Bebas! Semua aturan kecepatan dan larangan menyalip sudah selesai. Kembali ke aturan normal.";
        signDataMap.put("End of all speed and passing limits", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Wajib Belok Kanan";
        sign.explanationInIndonesian = "Di depan, semua mobil harus belok ke kanan ya.";
        signDataMap.put("Turn right ahead", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Wajib Belok Kiri";
        sign.explanationInIndonesian = "Semuanya siap-siap! Di depan kita harus belok ke kiri.";
        signDataMap.put("Turn left ahead", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Wajib Lurus";
        sign.explanationInIndonesian = "Lurus terus! Di sini tidak boleh belok kanan atau kiri.";
        signDataMap.put("Ahead only", sign);

        // Perbaikan nama key dari "Go straight or right" menjadi "Ahead or right" sesuai standar umum
        sign = new SignInfo();
        sign.nameInIndonesian = "Boleh Lurus atau Kanan";
        sign.explanationInIndonesian = "Kamu punya dua pilihan: boleh jalan lurus atau belok ke kanan.";
        signDataMap.put("Ahead or right", sign); // Key diperbaiki

        sign = new SignInfo();
        sign.nameInIndonesian = "Boleh Lurus atau Kiri";
        sign.explanationInIndonesian = "Di sini, kamu boleh memilih untuk jalan lurus atau belok ke kiri.";
        signDataMap.put("Ahead or left", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Tetap di Kanan";
        sign.explanationInIndonesian = "Ayo, semua mobil harus tetap berjalan di lajur sebelah kanan.";
        signDataMap.put("Keep right", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Tetap di Kiri";
        sign.explanationInIndonesian = "Semua mobil harus ambil lajur sebelah kiri ya.";
        signDataMap.put("Keep left", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Wajib Ikut Bundaran";
        sign.explanationInIndonesian = "Waktunya berputar! Ikuti jalan di bundaran ini dengan hati-hati.";
        signDataMap.put("Roundabout mandatory", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Akhir Dilarang Menyalip";
        sign.explanationInIndonesian = "Asiik! Sekarang mobil boleh menyalip lagi kalau aman.";
        signDataMap.put("End of no passing", sign);

        sign = new SignInfo();
        sign.nameInIndonesian = "Akhir Truk Dilarang Menyalip";
        sign.explanationInIndonesian = "Larangan menyalip untuk truk besar sudah selesai.";
        signDataMap.put("End of no passing by vehicles over 3.5 tons", sign);
    }

    // --- Sisa method helper (tidak berubah) ---
    private Bitmap toBitmap(ImageProxy image) {
        ImageProxy.PlaneProxy[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();
        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();
        byte[] nv21 = new byte[ySize + uSize + vSize];
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);
        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 100, out);
        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    private void loadTFLiteModel() {
        try {
            tflite = new Interpreter(loadModelFile());
            Log.d(TAG, "Model TFLite berhasil dimuat.");
        } catch (IOException e) {
            Log.e(TAG, "Error memuat model TFLite", e);
        }
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        try (AssetFileDescriptor fileDescriptor = getAssets().openFd("model_trained.tflite");
             FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }
    }

    private void loadLabels() {
        labels = new ArrayList<>();
        try (InputStream inputStream = getAssets().open("labels.txt");
             BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream))) {
            String line;
            while ((line = reader.readLine()) != null) {
                labels.add(line);
            }
            Log.d(TAG, "Labels berhasil dimuat: " + labels.size() + " label.");
        } catch (IOException e) {
            Log.e(TAG, "Error memuat labels.txt", e);
        }
    }

    private int getMaxIndex(float[] array) {
        if (array == null || array.length == 0) return -1;
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private void hideCameraViews() {
        if(permissionLayout.getVisibility() == View.GONE) {
            previewView.setVisibility(View.GONE);
            findViewById(R.id.camera_card).setVisibility(View.GONE);
            permissionLayout.setVisibility(View.VISIBLE);
        }
    }

    private void showCameraViews() {
        if(permissionLayout.getVisibility() == View.VISIBLE) {
            previewView.setVisibility(View.VISIBLE);
            findViewById(R.id.camera_card).setVisibility(View.VISIBLE);
            permissionLayout.setVisibility(View.GONE);
        }
    }

    private void openAppSettings() {
        Intent intent = new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS);
        Uri uri = Uri.fromParts("package", getPackageName(), null);
        intent.setData(uri);
        startActivity(intent);
    }
}