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
 * Analyzes image quality metrics: blur and brightness.
 */
object QualityAnalyzer {

    /**
     * Analyze quality of the region defined by bounding box.
     * @return QualityResult with blur and brightness scores.
     */
    fun analyze(bitmap: Bitmap, boundingBox: Rect): QualityResult {
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
            return QualityResult(0.0, 0.0, false, false)
        }

        val roi = Mat(mat, safeRect)

        // Convert to grayscale
        val gray = Mat()
        Imgproc.cvtColor(roi, gray, Imgproc.COLOR_RGBA2GRAY)

        // Blur score: Laplacian variance (higher = sharper)
        val laplacian = Mat()
        Imgproc.Laplacian(gray, laplacian, CvType.CV_64F)
        val stdDev = MatOfDouble()
        Core.meanStdDev(laplacian, MatOfDouble(), stdDev)
        val blurScore = stdDev.toArray()[0] * stdDev.toArray()[0]

        // Brightness: mean intensity
        val meanVal = Core.mean(gray)
        val brightness = meanVal.`val`[0]

        // Cleanup
        mat.release()
        roi.release()
        gray.release()
        laplacian.release()

        val blurOK = blurScore > 15.0
        val brightOK = brightness in 30.0..240.0

        return QualityResult(
            blurScore = blurScore,
            brightness = brightness,
            isSharp = blurOK,
            isWellLit = brightOK
        )
    }

    data class QualityResult(
        val blurScore: Double,
        val brightness: Double,
        val isSharp: Boolean,
        val isWellLit: Boolean
    ) {
        val isGood: Boolean get() = isSharp && isWellLit
    }
}
