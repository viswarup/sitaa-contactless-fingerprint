package com.grokking.contactlessfingerprint.core

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Rect
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.min

/**
 * ONNX-based liveness detector using MiniFASNet models.
 * Uses multiple models with different scale factors for robust detection.
 * 
 * Models:
 * - 2.7_80x80_MiniFASNetV2.onnx (scale 2.7)
 * - 4_0_0_80x80_MiniFASNetV1SE.onnx (scale 4.0)
 * 
 * Output classes:
 * - 0: Paper/Print (FAKE)
 * - 1: Real face (REAL)
 * - 2: Screen (FAKE)
 */
class OnnxLivenessDetector(private val context: Context) {
    
    companion object {
        private const val TAG = "OnnxLiveness"
        private const val INPUT_SIZE = 80
    }
    
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val sessions = mutableMapOf<String, OrtSession>()
    private var isLoaded = false
    
    init {
        loadModels()
    }
    
    /**
     * Load all ONNX models from assets.
     */
    private fun loadModels() {
        try {
            val modelFiles = listOf(
                "2.7_80x80_MiniFASNetV2.onnx",
                "4_0_0_80x80_MiniFASNetV1SE.onnx"
            )
            
            for (fileName in modelFiles) {
                try {
                    val bytes = context.assets.open(fileName).readBytes()
                    val session = env.createSession(bytes)
                    sessions[fileName] = session
                    Log.d(TAG, "✅ Loaded model: $fileName")
                } catch (e: Exception) {
                    Log.e(TAG, "❌ Failed to load model $fileName: ${e.message}")
                }
            }
            
            isLoaded = sessions.isNotEmpty()
            Log.d(TAG, "ONNX models loaded: ${sessions.size} models")
            
        } catch (e: Exception) {
            Log.e(TAG, "Error loading models: ${e.message}")
            e.printStackTrace()
        }
    }
    
    /**
     * Parse scale factor from model filename.
     * "2.7_80x80_MiniFASNetV2.onnx" -> 2.7
     * "4_0_0_80x80_MiniFASNetV1SE.onnx" -> 4.0
     */
    private fun parseScale(fileName: String): Float {
        val parts = fileName.split("_")
        return when {
            parts[0] == "4" -> 4.0f
            else -> parts[0].toFloatOrNull() ?: 2.7f
        }
    }
    
    /**
     * Expand bounding box using MiniFASNet scale expansion logic.
     * This is crucial for matching the training data preprocessing.
     */
    private fun expandBox(srcW: Int, srcH: Int, box: Rect, scale: Float): Rect {
        val bw = box.width().toFloat()
        val bh = box.height().toFloat()
        
        // Clamp scale to image bounds
        var s = min(scale, min((srcH - 1) / bh, (srcW - 1) / bw))
        
        val newW = bw * s
        val newH = bh * s
        
        val cx = box.left + bw / 2f
        val cy = box.top + bh / 2f
        
        var left = cx - newW / 2f
        var top = cy - newH / 2f
        var right = cx + newW / 2f
        var bottom = cy + newH / 2f
        
        // Fix boundaries
        if (left < 0) { right -= left; left = 0f }
        if (top < 0) { bottom -= top; top = 0f }
        if (right > srcW - 1) { left -= (right - (srcW - 1)); right = (srcW - 1).toFloat() }
        if (bottom > srcH - 1) { top -= (bottom - (srcH - 1)); bottom = (srcH - 1).toFloat() }
        
        // Ensure bounds are valid
        left = left.coerceIn(0f, (srcW - 1).toFloat())
        top = top.coerceIn(0f, (srcH - 1).toFloat())
        right = right.coerceIn(left + 1, srcW.toFloat())
        bottom = bottom.coerceIn(top + 1, srcH.toFloat())
        
        return Rect(left.toInt(), top.toInt(), right.toInt(), bottom.toInt())
    }
    
    /**
     * Preprocess image for ONNX model.
     * Crop -> Resize 80x80 -> BGR -> CHW layout -> FloatArray
     */
    private fun preprocess(bitmap: Bitmap, faceBox: Rect, scale: Float): FloatArray {
        val srcW = bitmap.width
        val srcH = bitmap.height
        
        // Expand bounding box
        val expanded = expandBox(srcW, srcH, faceBox, scale)
        
        // Crop face region
        val cropWidth = expanded.width().coerceAtLeast(1)
        val cropHeight = expanded.height().coerceAtLeast(1)
        val cropX = expanded.left.coerceIn(0, srcW - cropWidth)
        val cropY = expanded.top.coerceIn(0, srcH - cropHeight)
        
        val cropped = Bitmap.createBitmap(bitmap, cropX, cropY, cropWidth, cropHeight)
        
        // Resize to 80x80
        val resized = Bitmap.createScaledBitmap(cropped, INPUT_SIZE, INPUT_SIZE, true)
        
        // Convert to CHW format with BGR order (matching Python preprocessing)
        val chw = FloatArray(3 * INPUT_SIZE * INPUT_SIZE)
        var idx = 0
        
        // CHW layout: all B values, then all G values, then all R values
        for (c in listOf(2, 1, 0)) {  // BGR order
            for (y in 0 until INPUT_SIZE) {
                for (x in 0 until INPUT_SIZE) {
                    val pixel = resized.getPixel(x, y)
                    val value = when (c) {
                        0 -> Color.red(pixel)
                        1 -> Color.green(pixel)
                        2 -> Color.blue(pixel)
                        else -> 0
                    }
                    chw[idx++] = value.toFloat()
                }
            }
        }
        
        return chw
    }
    
    /**
     * Apply softmax to convert logits to probabilities.
     */
    private fun softmax(values: FloatArray): FloatArray {
        val maxVal = values.maxOrNull() ?: 0f
        val expVals = values.map { exp((it - maxVal).toDouble()) }
        val sum = expVals.sum()
        return expVals.map { (it / sum).toFloat() }.toFloatArray()
    }
    
    /**
     * Predict liveness for the given face.
     * 
     * @param bitmap Full camera frame
     * @param faceBox Bounding box from ML Kit face detection
     * @return LivenessResult with isReal flag and confidence
     */
    fun predict(bitmap: Bitmap, faceBox: Rect): LivenessResult {
        if (!isLoaded || sessions.isEmpty()) {
            Log.w(TAG, "Models not loaded, returning default")
            return LivenessResult(isReal = true, confidence = 0.5f, realScore = 0.5f)
        }
        
        try {
            // Fusion scores across all models: [paper, real, screen]
            val fusion = FloatArray(3) { 0f }
            
            for ((fileName, session) in sessions) {
                val scale = parseScale(fileName)
                val inputData = preprocess(bitmap, faceBox, scale)
                
                // Create input tensor [1, 3, 80, 80]
                val shape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
                val inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), shape)
                
                // Get input name
                val inputName = session.inputNames.firstOrNull() ?: "input"
                
                // Run inference
                val output = session.run(mapOf(inputName to inputTensor))
                
                // Get output tensor
                val outputTensor = output[0].value
                val scores = when (outputTensor) {
                    is Array<*> -> (outputTensor[0] as FloatArray)
                    else -> {
                        Log.w(TAG, "Unexpected output type: ${outputTensor?.javaClass}")
                        floatArrayOf(0.33f, 0.34f, 0.33f)
                    }
                }
                
                // Apply softmax
                val probs = softmax(scores)
                
                // Add to fusion
                for (i in 0..2) {
                    fusion[i] += probs[i]
                }
                
                Log.d(TAG, "Model $fileName: paper=${probs[0]}, real=${probs[1]}, screen=${probs[2]}")
                
                inputTensor.close()
                output.close()
            }
            
            // Average across models
            val numModels = sessions.size.toFloat()
            for (i in 0..2) {
                fusion[i] /= numModels
            }
            
            // Determine result
            // Label 0 = paper (FAKE), 1 = real (REAL), 2 = screen (FAKE)
            val maxIdx = fusion.indices.maxByOrNull { fusion[it] } ?: 1
            val isReal = (maxIdx == 1)
            val confidence = fusion[maxIdx]
            val realScore = fusion[1]
            
            Log.d(TAG, """
                === ONNX Liveness Result ===
                Paper (fake):  ${String.format("%.3f", fusion[0])}
                Real:          ${String.format("%.3f", fusion[1])}
                Screen (fake): ${String.format("%.3f", fusion[2])}
                Decision: ${if (isReal) "REAL" else "FAKE"} (confidence: ${String.format("%.3f", confidence)})
            """.trimIndent())
            
            return LivenessResult(isReal, confidence, realScore)
            
        } catch (e: Exception) {
            Log.e(TAG, "Prediction error: ${e.message}")
            e.printStackTrace()
            return LivenessResult(isReal = true, confidence = 0.5f, realScore = 0.5f)
        }
    }
    
    /**
     * Close all sessions and release resources.
     */
    fun close() {
        sessions.values.forEach { it.close() }
        sessions.clear()
        Log.d(TAG, "ONNX sessions closed")
    }
    
    /**
     * Result from liveness detection.
     */
    data class LivenessResult(
        val isReal: Boolean,
        val confidence: Float,
        val realScore: Float  // Score for "real" class specifically
    )
}
