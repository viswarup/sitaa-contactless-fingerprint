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
import kotlinx.coroutines.async
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
        var bitmap = CaptureActivity.capturedBitmap
        val fingerPoints = CaptureActivity.capturedFingerPoints

        if (bitmap == null) {
            binding.livenessText.text = "No image captured"
            Toast.makeText(this, "No image to process", Toast.LENGTH_SHORT).show()
            return
        }

        // OPTIMIZATION: Resize image if too large to speed up Gabor/Skeletonization
        // A width of ~600px is sufficient for fingerprint features and much faster
        val MAX_WIDTH = 600
        if (bitmap.width > MAX_WIDTH) {
            val ratio = MAX_WIDTH.toDouble() / bitmap.width
            val newHeight = (bitmap.height * ratio).toInt()
            bitmap = Bitmap.createScaledBitmap(bitmap, MAX_WIDTH, newHeight, true)
        }

        binding.livenessText.text = "Processing..."

        lifecycleScope.launch {
            // PART 1: Basic Enhancement (Steps 1-3) - Fast
            binding.livenessText.text = "Enhancing part 1..."
            
            val basicResult = withContext(Dispatchers.Default) {
                ImageEnhancer.enhanceBasic(bitmap!!, fingerPoints)
            }
            
            // Show Steps 1-3 immediately
            binding.step1Image.setImageBitmap(basicResult.step1_isolated)
            binding.step2Image.setImageBitmap(basicResult.step2_grayscale)
            binding.step3Image.setImageBitmap(basicResult.step3_clahe)
            
            // Show loading indicators for Steps 4-6
            binding.step4Image.setImageBitmap(null)
            binding.step5Image.setImageBitmap(null)
            binding.step6Image.setImageBitmap(null)
            binding.step4Progress.visibility = android.view.View.VISIBLE
            binding.step5Progress.visibility = android.view.View.VISIBLE
            binding.step6Progress.visibility = android.view.View.VISIBLE
            binding.step6Label.text = "Step 6: Minutiae Detection (Processing...)"

            // Check liveness in parallel with advanced enhancement
            val livenessDeferred = async(Dispatchers.Default) {
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

            // PART 2: Advanced Enhancement (Steps 4-6) - Slow
            binding.livenessText.text = "Enhancing part 2..."
            val finalResult = withContext(Dispatchers.Default) {
                ImageEnhancer.enhanceAdvanced(basicResult)
            }
            
            enhancementResult = finalResult
            
            // Hide progress bars and show results
            binding.step4Progress.visibility = android.view.View.GONE
            binding.step5Progress.visibility = android.view.View.GONE
            binding.step6Progress.visibility = android.view.View.GONE
            
            finalResult.step4_gabor?.let { binding.step4Image.setImageBitmap(it) }
            finalResult.step5_skeleton?.let { binding.step5Image.setImageBitmap(it) }
            finalResult.step6_minutiae?.let { binding.step6Image.setImageBitmap(it) }
            
            binding.step6Label.text = "Step 6: Minutiae Detection (${finalResult.minutiaeCount} found)"

            // Wait for liveness result
            val liveness = livenessDeferred.await()

            // ---------------------------------------------------------
            // LAYER 1: Heuristics (Texture, Color, Specular)
            // ---------------------------------------------------------
            binding.textureScoreText.text = "Texture: %.1f".format(liveness.textureScore)
            binding.colorScoreText.text = "Color: %.1f".format(liveness.colorVariance)
            binding.specularText.text = "Glare: %.1f%%".format(liveness.specularPercent)

            // ---------------------------------------------------------
            // LAYER 2: Motion Analysis
            // ---------------------------------------------------------
            val motion = CaptureActivity.motionResult
            if (motion != null) {
                binding.motionScoreText.text = "Score: %.2f".format(motion.motionScore)
                binding.motionVarianceText.text = "Var: %.3f".format(motion.motionVariance)
                
                binding.motionStatusText.text = if (motion.isLive) "PASS" else "FAIL"
                binding.motionStatusText.setTextColor(
                    if (motion.isLive) 0xFF4CAF50.toInt() else 0xFFF44336.toInt()
                )
            } else {
                binding.motionScoreText.text = "Score: N/A"
                binding.motionVarianceText.text = "Var: N/A"
                binding.motionStatusText.text = "N/A"
            }

            // ---------------------------------------------------------
            // LAYER 3: Deep Learning (ONNX)
            // ---------------------------------------------------------
            val onnxRes = CaptureActivity.onnxResult
            if (onnxRes != null) {
                binding.aiRealScoreText.text = "Real Prob: %.1f%%".format(onnxRes.realScore * 100)
                
                val statusText = if (onnxRes.isReal) "REAL" else "FAKE"
                binding.aiStatusText.text = "$statusText (${(onnxRes.confidence * 100).toInt()}%)"
                binding.aiStatusText.setTextColor(
                    if (onnxRes.isReal) 0xFF4CAF50.toInt() else 0xFFF44336.toInt()
                )
            } else {
                binding.aiRealScoreText.text = "Real Prob: --"
                binding.aiStatusText.text = "Not Run"
            }

            // ---------------------------------------------------------
            // Overall Decision
            // ---------------------------------------------------------
            val isHeuristicLive = liveness.isLive
            val isMotionLive = motion?.isLive == true
            val isAiLive = onnxRes?.isReal == true

            // Logic: Must pass AI + Motion + Heuristics
            val finalIsLive = isHeuristicLive && isMotionLive && isAiLive

            val status = if (finalIsLive) {
                "✓ LIVE FINGER DECTECTED"
            } else {
                val failReason = when {
                    !isAiLive -> "AI Detected Fake"
                    !isMotionLive -> "Motion Check Failed"
                    !isHeuristicLive -> liveness.reason // e.g. "Screen detected"
                    else -> "Verification Failed"
                }
                "⚠ $failReason"
            }
            
            binding.livenessText.text = status
            binding.livenessText.setTextColor(
                if (finalIsLive) 0xFF4CAF50.toInt() else 0xFFF44336.toInt()
            )
        }
    }

    private fun saveAllImages() {
        val result = enhancementResult ?: return
        val timestamp = System.currentTimeMillis()

        lifecycleScope.launch {
            var savedCount = 3
            withContext(Dispatchers.IO) {
                // Save Steps 1-3
                saveToGallery(result.step1_isolated, "finger_1_isolated_$timestamp.jpg")
                saveToGallery(result.step2_grayscale, "finger_2_grayscale_$timestamp.jpg")
                saveToGallery(result.step3_clahe, "finger_3_clahe_$timestamp.jpg")
                
                // Save Steps 4-6 if available
                result.step4_gabor?.let { 
                    saveToGallery(it, "finger_4_gabor_$timestamp.jpg")
                    savedCount++
                }
                result.step5_skeleton?.let { 
                    saveToGallery(it, "finger_5_skeleton_$timestamp.jpg")
                    savedCount++
                }
                result.step6_minutiae?.let { 
                    saveToGallery(it, "finger_6_minutiae_$timestamp.jpg")
                    savedCount++
                }
            }
            Toast.makeText(this@ResultActivity, "Saved $savedCount images to gallery", Toast.LENGTH_SHORT).show()
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
