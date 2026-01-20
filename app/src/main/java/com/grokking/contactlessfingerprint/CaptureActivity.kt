package com.grokking.contactlessfingerprint

import android.content.Intent
import android.graphics.Bitmap
import android.media.MediaActionSound
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.grokking.contactlessfingerprint.core.FingerDetector
import com.grokking.contactlessfingerprint.core.LivenessChecker
import com.grokking.contactlessfingerprint.core.MotionAnalyzer
import com.grokking.contactlessfingerprint.core.QualityAnalyzer
import com.grokking.contactlessfingerprint.databinding.ActivityCaptureBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class CaptureActivity : AppCompatActivity() {

    private lateinit var binding: ActivityCaptureBinding
    private lateinit var cameraExecutor: ExecutorService
    private var fingerDetector: FingerDetector? = null
    
    // Motion analyzer for liveness detection
    private val motionAnalyzer = MotionAnalyzer()

    // State
    private var goodFrameCount = 0
    private var isCountingDown = false
    private var countdownValue = 3
    private var hasCaptured = false
    private var lastCaptureBitmap: Bitmap? = null
    private var lastFingerPoints: List<Pair<Int, Int>> = emptyList()
    private var lastBoundingBox: android.graphics.Rect? = null

    private val handler = Handler(Looper.getMainLooper())
    private val STABILITY_THRESHOLD = 8

    companion object {
        private const val TAG = "CaptureActivity"
        var capturedBitmap: Bitmap? = null
        var capturedFingerPoints: List<Pair<Int, Int>> = emptyList()
        var motionResult: MotionAnalyzer.MotionResult? = null
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(this, "OpenCV failed to load", Toast.LENGTH_LONG).show()
            finish()
            return
        }

        binding = ActivityCaptureBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()

        fingerDetector = FingerDetector(this)
        if (!fingerDetector!!.isReady) {
            binding.debugText.text = "MediaPipe failed to initialize"
        }

        startCamera()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.previewView.surfaceProvider)
            }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        processFrame(imageProxy)
                    }
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageAnalyzer
                )
            } catch (e: Exception) {
                Log.e(TAG, "Camera binding failed", e)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun processFrame(imageProxy: ImageProxy) {
        if (hasCaptured) {
            imageProxy.close()
            return
        }

        val bitmap = imageProxy.toBitmap()
        imageProxy.close()

        val detection = fingerDetector?.detect(bitmap)

        if (detection == null) {
            runOnUiThread {
                updateUI(null, null, null)
            }
            return
        }

        val quality = QualityAnalyzer.analyze(bitmap, detection.boundingBox)
        val liveness = LivenessChecker.check(bitmap, detection.boundingBox)

        // Cache for capture
        lastCaptureBitmap = bitmap
        lastFingerPoints = detection.fingerPoints
        lastBoundingBox = detection.boundingBox

        // During countdown, collect frames for motion analysis
        if (isCountingDown && lastBoundingBox != null) {
            motionAnalyzer.addFrame(bitmap, lastBoundingBox!!)
        }

        runOnUiThread {
            updateUI(detection, quality, liveness)
        }
    }

    private fun updateUI(
        detection: FingerDetector.DetectionResult?,
        quality: QualityAnalyzer.QualityResult?,
        liveness: LivenessChecker.LivenessResult?
    ) {
        if (isFinishing || isDestroyed) return

        if (detection == null || quality == null) {
            binding.debugText.text = "No finger detected"
            binding.statusText.text = "Position your fingers"
            binding.qualityText.text = "Quality: --"
            cancelCountdown()
            return
        }

        val livenessText = if (liveness?.isLive == true) "LIVE" else "CHECK"
        binding.debugText.text = String.format(
            "Blur: %.0f | Bright: %.0f | %s (%.0f%%)",
            quality.blurScore, quality.brightness, livenessText, liveness?.confidence ?: 0.0
        )

        val color = when {
            quality.isGood && liveness?.isLive == true -> 0xFF4CAF50.toInt()
            quality.isGood -> 0xFFFFEB3B.toInt()
            else -> 0xFFF44336.toInt()
        }
        binding.debugText.setTextColor(color)

        binding.qualityText.text = String.format(
            "Width: %dpx | Sharp: %s | Lit: %s",
            detection.boundingBox.width(),
            if (quality.isSharp) "✓" else "✗",
            if (quality.isWellLit) "✓" else "✗"
        )

        val widthOK = detection.boundingBox.width() in 80..1000
        val allOK = widthOK && quality.isGood && (liveness?.isLive == true || (liveness?.confidence ?: 0.0) > 40)

        when {
            !widthOK -> {
                binding.statusText.text = "Move closer/farther"
                cancelCountdown()
            }
            !quality.isSharp -> {
                binding.statusText.text = "Hold steady"
                cancelCountdown()
            }
            !quality.isWellLit -> {
                binding.statusText.text = "Improve lighting"
                cancelCountdown()
            }
            liveness?.isLive != true && (liveness?.confidence ?: 0.0) <= 40 -> {
                binding.statusText.text = liveness?.reason ?: "Check liveness"
                cancelCountdown()
            }
            allOK -> {
                if (!isCountingDown) {
                    goodFrameCount++
                    binding.statusText.text = "Hold still... $goodFrameCount/$STABILITY_THRESHOLD"
                    
                    if (goodFrameCount >= STABILITY_THRESHOLD) {
                        startCountdown()
                    }
                }
            }
        }
    }

    private fun startCountdown() {
        if (isCountingDown) return
        isCountingDown = true
        countdownValue = 3
        goodFrameCount = 0
        
        // Reset motion analyzer for fresh analysis
        motionAnalyzer.reset()

        binding.statusText.text = "Don't move! Analyzing motion..."
        binding.countdownText.visibility = View.VISIBLE
        
        runCountdownStep()
    }

    private fun runCountdownStep() {
        if (!isCountingDown || isFinishing || isDestroyed) return

        if (countdownValue > 0) {
            binding.countdownText.text = countdownValue.toString()
            countdownValue--
            handler.postDelayed({ runCountdownStep() }, 1000)
        } else {
            binding.countdownText.visibility = View.GONE
            
            // Analyze motion before capture
            val motion = motionAnalyzer.analyze()
            motionResult = motion
            
            if (motion.isLive) {
                captureImage()
            } else {
                // Motion check failed - likely a screen or print
                binding.statusText.text = "⚠ ${motion.reason}"
                binding.statusText.setTextColor(0xFFF44336.toInt())
                
                // Reset and try again
                handler.postDelayed({
                    isCountingDown = false
                    motionAnalyzer.reset()
                    binding.statusText.setTextColor(0xFFFFFFFF.toInt())
                }, 2000)
            }
        }
    }

    private fun cancelCountdown() {
        if (isCountingDown) {
            isCountingDown = false
            handler.removeCallbacksAndMessages(null)
            binding.countdownText.visibility = View.GONE
            motionAnalyzer.reset()
        }
        goodFrameCount = 0
    }

    private fun captureImage() {
        if (hasCaptured || lastCaptureBitmap == null) return
        hasCaptured = true

        MediaActionSound().play(MediaActionSound.SHUTTER_CLICK)

        binding.statusText.text = "Captured! Processing..."

        capturedBitmap = lastCaptureBitmap
        capturedFingerPoints = lastFingerPoints

        lifecycleScope.launch {
            withContext(Dispatchers.IO) {
                saveBitmapToCache(lastCaptureBitmap!!)
            }
            
            startActivity(Intent(this@CaptureActivity, ResultActivity::class.java))
            finish()
        }
    }

    private fun saveBitmapToCache(bitmap: Bitmap) {
        val file = File(cacheDir, "captured_finger.jpg")
        FileOutputStream(file).use { out ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 95, out)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        fingerDetector?.close()
        handler.removeCallbacksAndMessages(null)
        motionAnalyzer.reset()
    }
}
