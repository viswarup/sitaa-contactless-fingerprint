package com.grokking.contactlessfingerprint

import android.content.ContentValues
import android.graphics.Bitmap
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.grokking.contactlessfingerprint.core.ImageEnhancer
import com.grokking.contactlessfingerprint.core.LivenessChecker
import com.grokking.contactlessfingerprint.databinding.ActivityResultBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

class ResultActivity : AppCompatActivity() {

    private lateinit var binding: ActivityResultBinding
    private var enhancementResult: ImageEnhancer.EnhancementResult? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityResultBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.retakeButton.setOnClickListener {
            finish()
        }

        binding.saveButton.setOnClickListener {
            saveAllImages()
        }

        processImage()
    }

    private fun processImage() {
        val bitmap = CaptureActivity.capturedBitmap
        val fingerPoints = CaptureActivity.capturedFingerPoints

        if (bitmap == null) {
            binding.livenessText.text = "No image captured"
            Toast.makeText(this, "No image to process", Toast.LENGTH_SHORT).show()
            return
        }

        binding.livenessText.text = "Processing..."

        lifecycleScope.launch {
            // Run enhancement pipeline
            val result = withContext(Dispatchers.Default) {
                ImageEnhancer.enhance(bitmap, fingerPoints)
            }

            enhancementResult = result

            // Display 3 enhancement steps (no Gabor)
            binding.step1Image.setImageBitmap(result.step1_isolated)
            binding.step2Image.setImageBitmap(result.step2_grayscale)
            binding.step3Image.setImageBitmap(result.step3_clahe)

            // Run liveness analysis with detailed scores
            val liveness = withContext(Dispatchers.Default) {
                if (fingerPoints.isNotEmpty()) {
                    val xs = fingerPoints.map { it.first }
                    val ys = fingerPoints.map { it.second }
                    val rect = android.graphics.Rect(
                        (xs.minOrNull()!! - 20).coerceAtLeast(0),
                        (ys.minOrNull()!! - 20).coerceAtLeast(0),
                        (xs.maxOrNull()!! + 20).coerceAtMost(bitmap.width),
                        (ys.maxOrNull()!! + 20).coerceAtMost(bitmap.height)
                    )
                    LivenessChecker.checkDetailed(bitmap, rect)
                } else {
                    LivenessChecker.DetailedResult(
                        isLive = false,
                        confidence = 0.0,
                        reason = "No finger points",
                        textureScore = 0.0,
                        colorVariance = 0.0,
                        specularPercent = 0.0
                    )
                }
            }

            // Update liveness UI
            val status = if (liveness.isLive) {
                "✓ LIVE FINGER (${liveness.confidence.toInt()}%)"
            } else {
                "⚠ ${liveness.reason}"
            }
            binding.livenessText.text = status
            binding.livenessText.setTextColor(
                if (liveness.isLive) 0xFF4CAF50.toInt() else 0xFFF44336.toInt()
            )

            // Show detailed scores
            binding.textureScoreText.text = "Texture: %.1f".format(liveness.textureScore)
            binding.colorScoreText.text = "Color: %.1f".format(liveness.colorVariance)
            binding.specularText.text = "Glare: %.1f%%".format(liveness.specularPercent)
            
            // Show motion analysis result
            val motion = CaptureActivity.motionResult
            if (motion != null) {
                binding.motionScoreText.text = "Motion: %.2f".format(motion.motionScore)
                binding.motionStatusText.text = if (motion.isLive) "✓ ${motion.reason}" else "⚠ ${motion.reason}"
                binding.motionStatusText.setTextColor(
                    if (motion.isLive) 0xFF4CAF50.toInt() else 0xFFF44336.toInt()
                )
            }
        }
    }

    private fun saveAllImages() {
        val result = enhancementResult ?: return
        val timestamp = System.currentTimeMillis()

        lifecycleScope.launch {
            withContext(Dispatchers.IO) {
                saveToGallery(result.step1_isolated, "finger_isolated_$timestamp.jpg")
                saveToGallery(result.step2_grayscale, "finger_grayscale_$timestamp.jpg")
                saveToGallery(result.step3_clahe, "finger_enhanced_$timestamp.jpg")
            }
            Toast.makeText(this@ResultActivity, "Saved 3 images to gallery", Toast.LENGTH_SHORT).show()
        }
    }

    private fun saveToGallery(bitmap: Bitmap, filename: String) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, filename)
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/Fingerprint")
            }

            val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            uri?.let {
                contentResolver.openOutputStream(it)?.use { out ->
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 95, out)
                }
            }
        } else {
            @Suppress("DEPRECATION")
            val dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
            val fingerDir = File(dir, "Fingerprint")
            fingerDir.mkdirs()
            val file = File(fingerDir, filename)
            FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 95, out)
            }
        }
    }
}
