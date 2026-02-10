package com.grokking.contactlessfingerprint.core

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc

/**
 * Enhances fingerprint images using:
 * 1. Convex hull segmentation
 * 2. Grayscale conversion
 * 3. CLAHE contrast enhancement
 */
object ImageEnhancer {

    data class EnhancementResult(
        val step1_isolated: Bitmap,
        val step2_grayscale: Bitmap,
        val step3_clahe: Bitmap,
        val step4_gabor: Bitmap?,        // Gabor enhanced
        val step5_skeleton: Bitmap?,     // Thinned ridges
        val step6_minutiae: Bitmap?,     // Skeleton with minutiae drawn
        val minutiaeCount: Int = 0
    )

    /**
     * Process image through the 6-stage enhancement pipeline.
     */
    /**
     * Stage 1: Basic Enhancement (Steps 1-3). Fast.
     */
    fun enhanceBasic(bitmap: Bitmap, fingerPoints: List<Pair<Int, Int>>): EnhancementResult {
        val original = Mat()
        Utils.bitmapToMat(bitmap, original)

        // Step 1: Convex Hull Segmentation
        val isolated = if (fingerPoints.isNotEmpty()) {
            isolateWithConvexHull(original, fingerPoints)
        } else {
            original.clone()
        }

        // Step 2: Convert to Grayscale
        val grayscale = Mat()
        Imgproc.cvtColor(isolated, grayscale, Imgproc.COLOR_RGBA2GRAY)

        // Step 3: CLAHE Enhancement
        val clahe = Imgproc.createCLAHE(2.5, Size(8.0, 8.0))
        val enhanced = Mat()
        clahe.apply(grayscale, enhanced)

        // Convert Steps 1-3 to bitmaps
        val isolatedBitmap = matToBitmap(isolated)
        val grayscaleBitmap = matToBitmap(grayscale)
        val enhancedBitmap = matToBitmap(enhanced)

        // Cleanup
        original.release()
        isolated.release()
        grayscale.release()
        enhanced.release()

        return EnhancementResult(
            step1_isolated = isolatedBitmap,
            step2_grayscale = grayscaleBitmap,
            step3_clahe = enhancedBitmap,
            step4_gabor = null,
            step5_skeleton = null,
            step6_minutiae = null,
            minutiaeCount = 0
        )
    }

    /**
     * Stage 2: Advanced Enhancement (Steps 4-6). Slow.
     * Takes the CLAHE image from Stage 1.
     */
    fun enhanceAdvanced(basicResult: EnhancementResult): EnhancementResult {
        // Steps 4-6: Advanced Pipeline (Gabor, Skeleton, Minutiae)
        val advancedResult = try {
            AdvancedEnhancer.process(basicResult.step3_clahe, null) // Mask is optional/null here
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }

        return basicResult.copy(
            step4_gabor = advancedResult?.gabor,
            step5_skeleton = advancedResult?.skeleton,
            step6_minutiae = advancedResult?.minutiaeImage,
            minutiaeCount = advancedResult?.minutiae?.size ?: 0
        )
    }

    // Keep old method for compatibility/reference if needed, or remove it.
    // Simplifying by removing the old monolithic function to force usage of new staged approach.

    private fun isolateWithConvexHull(src: Mat, points: List<Pair<Int, Int>>): Mat {
        val pointsMat = MatOfPoint()
        pointsMat.fromList(points.map { Point(it.first.toDouble(), it.second.toDouble()) })

        val hull = MatOfInt()
        Imgproc.convexHull(pointsMat, hull)

        val hullPoints = MatOfPoint()
        val hullIndices = hull.toArray()
        val sourcePoints = pointsMat.toArray()
        hullPoints.fromList(hullIndices.map { sourcePoints[it] })

        val mask = Mat.zeros(src.size(), CvType.CV_8UC1)
        Imgproc.fillConvexPoly(mask, hullPoints, Scalar(255.0))

        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(15.0, 15.0))
        Imgproc.dilate(mask, mask, kernel)

        val result = Mat()
        src.copyTo(result, mask)

        pointsMat.release()
        hull.release()
        hullPoints.release()
        mask.release()
        kernel.release()

        return result
    }

    private fun matToBitmap(mat: Mat): Bitmap {
        val displayMat = if (mat.channels() == 1) {
            val colored = Mat()
            Imgproc.cvtColor(mat, colored, Imgproc.COLOR_GRAY2RGBA)
            colored
        } else if (mat.channels() == 3) {
            val colored = Mat()
            Imgproc.cvtColor(mat, colored, Imgproc.COLOR_RGB2RGBA)
            colored
        } else {
            mat
        }

        val bitmap = Bitmap.createBitmap(displayMat.cols(), displayMat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(displayMat, bitmap)
        
        if (displayMat != mat) displayMat.release()
        
        return bitmap
    }
}
