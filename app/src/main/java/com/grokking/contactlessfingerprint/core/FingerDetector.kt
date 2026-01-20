package com.grokking.contactlessfingerprint.core

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker

/**
 * Simple MediaPipe-based finger detector.
 * Uses landmarks 6-8, 10-12, 14-16, 18-20 (finger segments only, not palm).
 */
class FingerDetector(context: Context) {

    private var handLandmarker: HandLandmarker? = null
    var isReady = false
        private set

    init {
        try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("hand_landmarker.task")
                .build()

            val options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.IMAGE)
                .setNumHands(1)
                .setMinHandDetectionConfidence(0.3f)
                .setMinHandPresenceConfidence(0.3f)
                .setMinTrackingConfidence(0.3f)
                .build()

            handLandmarker = HandLandmarker.createFromOptions(context, options)
            isReady = true
            Log.d(TAG, "✓ FingerDetector initialized")
        } catch (e: Exception) {
            Log.e(TAG, "✗ Failed to initialize: ${e.message}")
            isReady = false
        }
    }

    /**
     * Detect fingers in bitmap.
     * @return DetectionResult with bounding box and landmark points, or null if no hand found.
     */
    fun detect(bitmap: Bitmap): DetectionResult? {
        if (!isReady || handLandmarker == null) return null

        val mpImage = BitmapImageBuilder(bitmap).build()

        val result = try {
            handLandmarker!!.detect(mpImage)
        } catch (e: Exception) {
            Log.e(TAG, "Detection error: ${e.message}")
            return null
        }

        if (result.landmarks().isEmpty()) {
            return null
        }

        val landmarks = result.landmarks()[0]
        if (landmarks.size < 21) return null

        // Use finger segment landmarks only (not palm/wrist)
        // Index: 6,7,8 | Middle: 10,11,12 | Ring: 14,15,16 | Pinky: 18,19,20
        val fingerIndices = listOf(6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20)
        val fingerLandmarks = fingerIndices.map { landmarks[it] }

        // Calculate bounding box
        val xs = fingerLandmarks.map { (it.x() * bitmap.width).toInt() }
        val ys = fingerLandmarks.map { (it.y() * bitmap.height).toInt() }

        val padding = 30
        val boundingBox = Rect(
            (xs.minOrNull()!! - padding).coerceAtLeast(0),
            (ys.minOrNull()!! - padding).coerceAtLeast(0),
            (xs.maxOrNull()!! + padding).coerceAtMost(bitmap.width),
            (ys.maxOrNull()!! + padding).coerceAtMost(bitmap.height)
        )

        // Convert to pixel coordinates for convex hull
        val fingerPoints = fingerLandmarks.map {
            Pair((it.x() * bitmap.width).toInt(), (it.y() * bitmap.height).toInt())
        }

        val confidence = result.handednesses()[0][0].score()
        Log.d(TAG, "✓ Hand detected: ${fingerPoints.size} points, conf=${(confidence*100).toInt()}%")

        return DetectionResult(
            boundingBox = boundingBox,
            fingerPoints = fingerPoints,
            confidence = confidence
        )
    }

    fun close() {
        handLandmarker?.close()
    }

    data class DetectionResult(
        val boundingBox: Rect,
        val fingerPoints: List<Pair<Int, Int>>,
        val confidence: Float
    )

    companion object {
        private const val TAG = "FingerDetector"
    }
}
