package com.grokking.contactlessfingerprint.core

import android.graphics.Bitmap
import android.graphics.Rect
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

/**
 * Analyzes lighting conditions to detect backlighting or low light.
 */
object LightingAnalyzer {

    data class LightingResult(
        val needsFlash: Boolean,
        val reason: String = ""
    )

    /**
     * Analyze lighting based on the full image and the finger ROI.
     * @param bitmap The full frame bitmap.
     * @param fingerBoundingBox The bounding box of the detected finger.
     */
    fun analyze(bitmap: Bitmap, fingerBoundingBox: Rect): LightingResult {
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)

        // Convert to grayscale for intensity analysis
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGBA2GRAY)

        // 1. Analyze Finger ROI Brightness
        // Ensure ROI is within bounds
        val safeRoi = org.opencv.core.Rect(
            fingerBoundingBox.left.coerceAtLeast(0),
            fingerBoundingBox.top.coerceAtLeast(0),
            fingerBoundingBox.width().coerceAtMost(gray.cols() - fingerBoundingBox.left.coerceAtLeast(0)),
            fingerBoundingBox.height().coerceAtMost(gray.rows() - fingerBoundingBox.top.coerceAtLeast(0))
        )

        // If ROI is invalid, we can't judge finger brightness, but we can check global low light
        if (safeRoi.width <= 0 || safeRoi.height <= 0) {
            val globalMean = Core.mean(gray).`val`[0]
            val isLowLight = globalMean < 60.0 // Threshold for very dark scene
            
            // Clean up
            mat.release()
            gray.release()
            
            return if (isLowLight) {
                LightingResult(true, "Low Light")
            } else {
                LightingResult(false)
            }
        }

        val fingerMat = Mat(gray, safeRoi)
        val fingerBrightness = Core.mean(fingerMat).`val`[0]
        val globalBrightness = Core.mean(gray).`val`[0]

        // Clean up immediately
        fingerMat.release()
        gray.release()
        mat.release()

        // Thresholds
        val LOW_LIGHT_THRESHOLD = 80.0
        val BACKLIGHT_BG_THRESHOLD = 150.0 // Bright background
        val BACKLIGHT_FINGER_THRESHOLD = 100.0 // Dark finger relative to background

        // Logic
        // 1. Low Light: Everything is dark
        if (fingerBrightness < LOW_LIGHT_THRESHOLD && globalBrightness < LOW_LIGHT_THRESHOLD) {
            return LightingResult(true, "Low Light")
        }

        // 2. Backlighting: Bright background, dark finger
        if (globalBrightness > BACKLIGHT_BG_THRESHOLD && fingerBrightness < BACKLIGHT_FINGER_THRESHOLD) {
            return LightingResult(true, "Backlighting")
        }

        return LightingResult(false)
    }
}
