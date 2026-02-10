package com.grokking.contactlessfingerprint.core

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.*

/**
 * Advanced fingerprint enhancement pipeline:
 * 1. Gabor Filter Bank (16 orientations) for ridge enhancement
 * 2. Zhang-Suen Thinning for skeletonization
 * 3. Crossing Number method for minutiae extraction
 * 4. Spurious minutiae filtering
 */
object AdvancedEnhancer {

    data class Minutia(
        val x: Int,
        val y: Int,
        val type: MinutiaType,
        val direction: Double,
        val quality: Double = 1.0
    )

    enum class MinutiaType {
        ENDING,      // Ridge ending (CN = 1)
        BIFURCATION  // Ridge bifurcation (CN = 3)
    }

    data class AdvancedResult(
        val gabor: Bitmap,
        val skeleton: Bitmap,
        val minutiaeImage: Bitmap,
        val minutiae: List<Minutia>
    )

    /**
     * Run the advanced pipeline on a CLAHE-enhanced grayscale image.
     */
    fun process(claheImage: Bitmap, mask: Bitmap?): AdvancedResult {
        val grayMat = Mat()
        Utils.bitmapToMat(claheImage, grayMat)
        
        // Convert to single channel if needed
        val gray = if (grayMat.channels() > 1) {
            val single = Mat()
            Imgproc.cvtColor(grayMat, single, Imgproc.COLOR_RGBA2GRAY)
            grayMat.release()
            single
        } else {
            grayMat
        }

        // Load mask if provided
        val maskMat: Mat? = mask?.let {
            val m = Mat()
            Utils.bitmapToMat(it, m)
            if (m.channels() > 1) {
                val single = Mat()
                Imgproc.cvtColor(m, single, Imgproc.COLOR_RGBA2GRAY)
                m.release()
                single
            } else {
                m
            }
        }

        // Step 1: Gabor Filter Bank
        val gabor = applyGaborFilter(gray)

        // Step 2: Binarize and Skeletonize
        val binary = Mat()
        Imgproc.adaptiveThreshold(gabor, binary, 255.0,
            Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 15, -2.0)
        
        // Apply mask if available
        maskMat?.let {
            Core.bitwise_and(binary, it, binary)
        }

        val skeleton = skeletonize(binary)

        // Step 3: Extract Minutiae
        val rawMinutiae = extractMinutiae(skeleton, maskMat)

        // Step 4: Filter Spurious Minutiae
        val filteredMinutiae = if (maskMat != null) {
            filterSpuriousMinutiae(rawMinutiae, maskMat)
        } else {
            rawMinutiae.take(50)
        }

        // Step 5: Draw Minutiae on skeleton
        val minutiaeVis = drawMinutiae(skeleton, filteredMinutiae)

        // Convert to bitmaps
        val gaborBitmap = matToBitmap(gabor)
        val skeletonBitmap = matToBitmap(skeleton)
        val minutiaeBitmap = matToBitmap(minutiaeVis)

        // Cleanup
        gray.release()
        gabor.release()
        binary.release()
        skeleton.release()
        minutiaeVis.release()
        maskMat?.release()

        return AdvancedResult(gaborBitmap, skeletonBitmap, minutiaeBitmap, filteredMinutiae)
    }

    /**
     * 16-orientation Gabor filter bank optimized for contactless fingerprints.
     */
    private fun applyGaborFilter(gray: Mat): Mat {
        val accumulator = Mat.zeros(gray.size(), CvType.CV_32F)

        // 16 orientations from 0 to π
        for (i in 0 until 16) {
            val theta = i * PI / 16

            // Gabor parameters optimized for contactless fingerprints:
            // ksize: 31x31 for larger ridges
            // sigma: 5.0 for smooth response
            // lambd: 10.0 (wavelength ~10 pixels typical)
            // gamma: 0.5 (aspect ratio)
            val kernel = Imgproc.getGaborKernel(
                Size(31.0, 31.0),
                5.0,    // sigma
                theta,  // theta
                10.0,   // lambda
                0.5,    // gamma
                0.0,    // psi
                CvType.CV_32F
            )

            val filtered = Mat()
            Imgproc.filter2D(gray, filtered, CvType.CV_32F, kernel)

            // Take maximum response across all orientations
            Core.max(accumulator, filtered, accumulator)

            kernel.release()
            filtered.release()
        }

        // Normalize to 0-255
        val result = Mat()
        Core.normalize(accumulator, result, 0.0, 255.0, Core.NORM_MINMAX)
        result.convertTo(result, CvType.CV_8UC1)

        accumulator.release()
        return result
    }

    /**
     * Zhang-Suen thinning algorithm for skeletonization.
     * Reduces binary ridges to 1-pixel wide lines.
     */
    private fun skeletonize(binary: Mat): Mat {
        val skeleton = binary.clone()
        
        // Ensure binary values are 0 and 1
        val normalized = Mat()
        skeleton.convertTo(normalized, CvType.CV_8UC1)
        Core.divide(normalized, Scalar(255.0), normalized)

        var changing = true
        while (changing) {
            changing = false

            // Sub-iteration 1
            val toRemove1 = mutableListOf<Point>()
            for (y in 1 until normalized.rows() - 1) {
                for (x in 1 until normalized.cols() - 1) {
                    if (normalized.get(y, x)[0] != 1.0) continue
                    
                    val neighbors = getNeighbors(normalized, x, y)
                    val p2 = neighbors[0]; val p3 = neighbors[1]
                    val p4 = neighbors[2]; val p5 = neighbors[3]
                    val p6 = neighbors[4]; val p7 = neighbors[5]
                    val p8 = neighbors[6]; val p9 = neighbors[7]

                    val B = neighbors.sum()
                    val A = countTransitions(neighbors)

                    if (B in 2.0..6.0 && A == 1 &&
                        p2 * p4 * p6 == 0.0 && p4 * p6 * p8 == 0.0) {
                        toRemove1.add(Point(x.toDouble(), y.toDouble()))
                        changing = true
                    }
                }
            }
            for (p in toRemove1) {
                normalized.put(p.y.toInt(), p.x.toInt(), 0.0)
            }

            // Sub-iteration 2
            val toRemove2 = mutableListOf<Point>()
            for (y in 1 until normalized.rows() - 1) {
                for (x in 1 until normalized.cols() - 1) {
                    if (normalized.get(y, x)[0] != 1.0) continue

                    val neighbors = getNeighbors(normalized, x, y)
                    val p2 = neighbors[0]; val p3 = neighbors[1]
                    val p4 = neighbors[2]; val p5 = neighbors[3]
                    val p6 = neighbors[4]; val p7 = neighbors[5]
                    val p8 = neighbors[6]; val p9 = neighbors[7]

                    val B = neighbors.sum()
                    val A = countTransitions(neighbors)

                    if (B in 2.0..6.0 && A == 1 &&
                        p2 * p4 * p8 == 0.0 && p2 * p6 * p8 == 0.0) {
                        toRemove2.add(Point(x.toDouble(), y.toDouble()))
                        changing = true
                    }
                }
            }
            for (p in toRemove2) {
                normalized.put(p.y.toInt(), p.x.toInt(), 0.0)
            }
        }

        // Convert back to 0-255
        Core.multiply(normalized, Scalar(255.0), normalized)
        normalized.convertTo(skeleton, CvType.CV_8UC1)
        normalized.release()

        return skeleton
    }

    private fun getNeighbors(mat: Mat, x: Int, y: Int): DoubleArray {
        // Clockwise from top: P2, P3, P4, P5, P6, P7, P8, P9
        return doubleArrayOf(
            mat.get(y - 1, x)[0],     // P2 (N)
            mat.get(y - 1, x + 1)[0], // P3 (NE)
            mat.get(y, x + 1)[0],     // P4 (E)
            mat.get(y + 1, x + 1)[0], // P5 (SE)
            mat.get(y + 1, x)[0],     // P6 (S)
            mat.get(y + 1, x - 1)[0], // P7 (SW)
            mat.get(y, x - 1)[0],     // P8 (W)
            mat.get(y - 1, x - 1)[0]  // P9 (NW)
        )
    }

    private fun countTransitions(neighbors: DoubleArray): Int {
        var count = 0
        for (i in 0 until 8) {
            if (neighbors[i] == 0.0 && neighbors[(i + 1) % 8] == 1.0) {
                count++
            }
        }
        return count
    }

    /**
     * Extract minutiae using the Crossing Number method.
     */
    private fun extractMinutiae(skeleton: Mat, mask: Mat?): List<Minutia> {
        val minutiae = mutableListOf<Minutia>()

        for (y in 1 until skeleton.rows() - 1) {
            for (x in 1 until skeleton.cols() - 1) {
                if (skeleton.get(y, x)[0] < 128) continue

                // Check mask
                if (mask != null && mask.get(y, x)[0] < 128) continue

                // Get 8-connected neighbors
                val neighbors = intArrayOf(
                    if (skeleton.get(y - 1, x)[0] > 128) 1 else 0,
                    if (skeleton.get(y - 1, x + 1)[0] > 128) 1 else 0,
                    if (skeleton.get(y, x + 1)[0] > 128) 1 else 0,
                    if (skeleton.get(y + 1, x + 1)[0] > 128) 1 else 0,
                    if (skeleton.get(y + 1, x)[0] > 128) 1 else 0,
                    if (skeleton.get(y + 1, x - 1)[0] > 128) 1 else 0,
                    if (skeleton.get(y, x - 1)[0] > 128) 1 else 0,
                    if (skeleton.get(y - 1, x - 1)[0] > 128) 1 else 0
                )

                // Crossing number
                var cn = 0
                for (i in 0 until 8) {
                    cn += abs(neighbors[i] - neighbors[(i + 1) % 8])
                }
                cn /= 2

                val direction = calculateDirection(skeleton, x, y)

                when (cn) {
                    1 -> minutiae.add(Minutia(x, y, MinutiaType.ENDING, direction))
                    3 -> minutiae.add(Minutia(x, y, MinutiaType.BIFURCATION, direction))
                }
            }
        }

        return minutiae
    }

    private fun calculateDirection(skeleton: Mat, x: Int, y: Int): Double {
        val half = 3
        var sumX = 0.0
        var sumY = 0.0
        var count = 0

        for (dy in -half..half) {
            for (dx in -half..half) {
                val ny = y + dy
                val nx = x + dx
                if (ny in 0 until skeleton.rows() && nx in 0 until skeleton.cols()) {
                    if (skeleton.get(ny, nx)[0] > 128) {
                        sumX += dx
                        sumY += dy
                        count++
                    }
                }
            }
        }

        return if (count > 0) atan2(sumY, sumX) else 0.0
    }

    /**
     * Filter spurious minutiae based on border distance and clustering.
     */
    private fun filterSpuriousMinutiae(
        minutiae: List<Minutia>,
        mask: Mat,
        minBorderDist: Int = 15,
        minClusterDist: Int = 10,
        maxMinutiae: Int = 50
    ): List<Minutia> {
        val h = mask.rows()
        val w = mask.cols()

        // Step 1: Filter by border distance
        val borderFiltered = minutiae.filter { m ->
            // Image border
            if (m.x < minBorderDist || m.x > w - minBorderDist) return@filter false
            if (m.y < minBorderDist || m.y > h - minBorderDist) return@filter false

            // Mask border (check local region for zeros)
            val y1 = (m.y - minBorderDist).coerceAtLeast(0)
            val y2 = (m.y + minBorderDist).coerceAtMost(h)
            val x1 = (m.x - minBorderDist).coerceAtLeast(0)
            val x2 = (m.x + minBorderDist).coerceAtMost(w)

            for (cy in y1 until y2) {
                for (cx in x1 until x2) {
                    if (mask.get(cy, cx)[0] < 128) {
                        // Near mask border
                        val dist = sqrt(((m.x - cx) * (m.x - cx) + (m.y - cy) * (m.y - cy)).toDouble())
                        if (dist < minBorderDist) return@filter false
                    }
                }
            }
            true
        }.toMutableList()

        // Step 2: Remove clustered minutiae
        val toRemove = mutableSetOf<Int>()
        for (i in borderFiltered.indices) {
            if (i in toRemove) continue
            for (j in i + 1 until borderFiltered.size) {
                if (j in toRemove) continue
                val dx = borderFiltered[i].x - borderFiltered[j].x
                val dy = borderFiltered[i].y - borderFiltered[j].y
                val dist = sqrt((dx * dx + dy * dy).toDouble())
                if (dist < minClusterDist) {
                    // Remove lower quality one
                    if (borderFiltered[i].quality < borderFiltered[j].quality) {
                        toRemove.add(i)
                    } else {
                        toRemove.add(j)
                    }
                }
            }
        }

        val clusterFiltered = borderFiltered.filterIndexed { index, _ -> index !in toRemove }

        // Step 3: Keep top N by quality
        return clusterFiltered
            .sortedByDescending { it.quality }
            .take(maxMinutiae)
    }

    /**
     * Draw minutiae on the skeleton image.
     * Red circles for endings, blue circles for bifurcations.
     */
    private fun drawMinutiae(skeleton: Mat, minutiae: List<Minutia>): Mat {
        val vis = Mat()
        Imgproc.cvtColor(skeleton, vis, Imgproc.COLOR_GRAY2BGR)

        for (m in minutiae) {
            val color = when (m.type) {
                MinutiaType.ENDING -> Scalar(0.0, 0.0, 255.0)      // Red
                MinutiaType.BIFURCATION -> Scalar(255.0, 0.0, 0.0) // Blue
            }
            Imgproc.circle(vis, Point(m.x.toDouble(), m.y.toDouble()), 5, color, 2)
        }

        return vis
    }

    private fun matToBitmap(mat: Mat): Bitmap {
        val displayMat = when {
            mat.channels() == 1 -> {
                val colored = Mat()
                Imgproc.cvtColor(mat, colored, Imgproc.COLOR_GRAY2RGBA)
                colored
            }
            mat.channels() == 3 -> {
                val colored = Mat()
                Imgproc.cvtColor(mat, colored, Imgproc.COLOR_BGR2RGBA)
                colored
            }
            else -> mat
        }

        val bitmap = Bitmap.createBitmap(displayMat.cols(), displayMat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(displayMat, bitmap)

        if (displayMat != mat) displayMat.release()

        return bitmap
    }
}
