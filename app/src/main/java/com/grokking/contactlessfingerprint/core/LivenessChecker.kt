package com.grokking.contactlessfingerprint.core

import android.graphics.Bitmap
import android.graphics.Rect
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.imgproc.Imgproc

/**
 * Checks if a finger image is from a live finger (vs photo/printout/screen).
 * Uses multiple methods:
 * 1. Texture analysis (Laplacian variance)
 * 2. Color variance (HSV saturation)
 * 3. Specular reflection detection (screen glare/hotspots)
 */
object LivenessChecker {

    private const val TEXTURE_THRESHOLD = 20.0
    private const val COLOR_THRESHOLD = 6.0
    private const val SPECULAR_THRESHOLD = 0.02  // Max percentage of very bright pixels

    fun check(bitmap: Bitmap, boundingBox: Rect): LivenessResult {
        val detailed = checkDetailed(bitmap, boundingBox)
        return LivenessResult(detailed.isLive, detailed.confidence, detailed.reason)
    }

    fun checkDetailed(bitmap: Bitmap, boundingBox: Rect): DetailedResult {
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)

        val safeRect = org.opencv.core.Rect(
            boundingBox.left.coerceAtLeast(0),
            boundingBox.top.coerceAtLeast(0),
            boundingBox.width().coerceAtMost(mat.cols() - boundingBox.left.coerceAtLeast(0)),
            boundingBox.height().coerceAtMost(mat.rows() - boundingBox.top.coerceAtLeast(0))
        )

        if (safeRect.width <= 0 || safeRect.height <= 0) {
            mat.release()
            return DetailedResult(false, 0.0, "Invalid region", 0.0, 0.0, 0.0)
        }

        val roi = Mat(mat, safeRect)

        // 1. Texture analysis: Laplacian variance
        val gray = Mat()
        Imgproc.cvtColor(roi, gray, Imgproc.COLOR_RGBA2GRAY)
        val laplacian = Mat()
        Imgproc.Laplacian(gray, laplacian, CvType.CV_64F)
        val textureStdDev = MatOfDouble()
        Core.meanStdDev(laplacian, MatOfDouble(), textureStdDev)
        val textureScore = textureStdDev.toArray()[0] * textureStdDev.toArray()[0]

        // 2. Color variance: Saturation std dev
        val rgb = Mat()
        val hsv = Mat()
        Imgproc.cvtColor(roi, rgb, Imgproc.COLOR_RGBA2RGB)
        Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)
        val channels = ArrayList<Mat>()
        Core.split(hsv, channels)
        val satStdDev = MatOfDouble()
        Core.meanStdDev(channels[1], MatOfDouble(), satStdDev)
        val colorVariance = satStdDev.toArray()[0]

        // 3. Specular reflection detection (screens have bright hotspots)
        // Count pixels with very high brightness (>250)
        val totalPixels = gray.rows() * gray.cols()
        val brightMask = Mat()
        Imgproc.threshold(gray, brightMask, 250.0, 255.0, Imgproc.THRESH_BINARY)
        val brightPixels = Core.countNonZero(brightMask)
        val specularRatio = brightPixels.toDouble() / totalPixels
        
        // Cleanup
        mat.release()
        roi.release()
        gray.release()
        laplacian.release()
        rgb.release()
        hsv.release()
        channels.forEach { it.release() }
        brightMask.release()

        // Decision logic
        val textureOK = textureScore > TEXTURE_THRESHOLD
        val colorOK = colorVariance > COLOR_THRESHOLD
        val noSpecular = specularRatio < SPECULAR_THRESHOLD  // Low specular = no screen glare

        val isLive = textureOK && colorOK && noSpecular
        
        // Confidence calculation
        val textureConf = (textureScore / 80.0).coerceIn(0.0, 1.0) * 35
        val colorConf = (colorVariance / 25.0).coerceIn(0.0, 1.0) * 35
        val specularConf = if (noSpecular) 30.0 else 0.0
        val confidence = (textureConf + colorConf + specularConf).coerceIn(0.0, 100.0)

        val reason = when {
            !noSpecular -> "Screen detected (glare)"
            !textureOK && !colorOK -> "Flat image (print/photo)"
            !textureOK -> "Low texture (possible print)"
            !colorOK -> "Uniform color (possible fake)"
            else -> "Live finger detected"
        }

        return DetailedResult(isLive, confidence, reason, textureScore, colorVariance, specularRatio * 100)
    }

    data class LivenessResult(
        val isLive: Boolean,
        val confidence: Double,
        val reason: String
    )

    data class DetailedResult(
        val isLive: Boolean,
        val confidence: Double,
        val reason: String,
        val textureScore: Double,
        val colorVariance: Double,
        val specularPercent: Double
    )
}
