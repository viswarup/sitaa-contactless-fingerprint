package com.grokking.contactlessfingerprint.core

import android.graphics.Bitmap
import android.graphics.Rect
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.video.Video

/**
 * Analyzes motion between frames to detect liveness.
 * Real fingers have natural micro-tremor from hand movement.
 * Screens/prints are perfectly still.
 */
class MotionAnalyzer {
    
    private val frameBuffer = mutableListOf<Mat>()
    private val maxFrames = 8  // Keep last 8 frames
    
    companion object {
        private const val MOTION_THRESHOLD = 0.5  // Minimum average motion for live finger
        private const val MAX_MOTION = 15.0       // Maximum motion (too much = shaking)
    }
    
    /**
     * Add a frame to the buffer for motion analysis.
     * Call this for each frame during countdown.
     */
    fun addFrame(bitmap: Bitmap, boundingBox: Rect) {
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)
        
        // Create safe ROI
        val safeRect = org.opencv.core.Rect(
            boundingBox.left.coerceAtLeast(0),
            boundingBox.top.coerceAtLeast(0),
            boundingBox.width().coerceAtMost(mat.cols() - boundingBox.left.coerceAtLeast(0)),
            boundingBox.height().coerceAtMost(mat.rows() - boundingBox.top.coerceAtLeast(0))
        )
        
        if (safeRect.width <= 0 || safeRect.height <= 0) {
            mat.release()
            return
        }
        
        val roi = Mat(mat, safeRect)
        
        // Convert to grayscale for optical flow
        val gray = Mat()
        Imgproc.cvtColor(roi, gray, Imgproc.COLOR_RGBA2GRAY)
        
        // Resize for faster processing
        val resized = Mat()
        Imgproc.resize(gray, resized, Size(100.0, 150.0))
        
        // Add to buffer
        frameBuffer.add(resized)
        
        // Keep only last N frames
        while (frameBuffer.size > maxFrames) {
            frameBuffer.removeAt(0).release()
        }
        
        // Cleanup
        mat.release()
        roi.release()
        gray.release()
    }
    
    /**
     * Analyze motion across stored frames.
     * Returns MotionResult with motion score and liveness decision.
     */
    fun analyze(): MotionResult {
        if (frameBuffer.size < 3) {
            return MotionResult(
                isLive = false,
                motionScore = 0.0,
                reason = "Not enough frames (${frameBuffer.size})"
            )
        }
        
        val motionScores = mutableListOf<Double>()
        
        // Calculate motion between consecutive frame pairs
        for (i in 1 until frameBuffer.size) {
            val prevFrame = frameBuffer[i - 1]
            val currFrame = frameBuffer[i]
            
            val motion = calculateFrameDifference(prevFrame, currFrame)
            motionScores.add(motion)
        }
        
        // Calculate average and variance of motion
        val avgMotion = motionScores.average()
        val motionVariance = motionScores.map { (it - avgMotion) * (it - avgMotion) }.average()
        
        // Decision logic:
        // - Real fingers: small but consistent motion (0.5-15 avg, some variance)
        // - Screens/prints: near-zero motion (< 0.5)
        // - Shaky hands: high motion (> 15)
        
        val hasMotion = avgMotion > MOTION_THRESHOLD
        val notTooShaky = avgMotion < MAX_MOTION
        val hasVariance = motionVariance > 0.1  // Real motion has some variance
        
        val isLive = hasMotion && notTooShaky && hasVariance
        
        val reason = when {
            avgMotion < MOTION_THRESHOLD -> "No movement detected (static image)"
            avgMotion > MAX_MOTION -> "Too much movement (hold steady)"
            !hasVariance -> "Uniform motion (possible screen)"
            else -> "Natural micro-tremor detected"
        }
        
        return MotionResult(
            isLive = isLive,
            motionScore = avgMotion,
            reason = reason,
            motionVariance = motionVariance,
            frameCount = frameBuffer.size
        )
    }
    
    /**
     * Calculate motion between two frames using frame difference.
     */
    private fun calculateFrameDifference(prev: Mat, curr: Mat): Double {
        val diff = Mat()
        Core.absdiff(prev, curr, diff)
        
        // Calculate mean of absolute differences
        val mean = Core.mean(diff)
        diff.release()
        
        return mean.`val`[0]
    }
    
    /**
     * Clear the frame buffer.
     */
    fun reset() {
        frameBuffer.forEach { it.release() }
        frameBuffer.clear()
    }
    
    data class MotionResult(
        val isLive: Boolean,
        val motionScore: Double,
        val reason: String,
        val motionVariance: Double = 0.0,
        val frameCount: Int = 0
    )
}
