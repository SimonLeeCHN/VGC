# VGC and downstream indentation measurement

This document explains the downstream measurement methods used in **Unsupervised Gray-Level Compression for Improved Automatic Measurement of Vickers Indentations**, and how VGC is incorporated into their measurement pipelines.

## Role of VGC

Vickers Gray-level Compression (VGC) is a grayscale-image preprocessing method implemented using PyTorch. It optimizes a network separately for each input image, without manually segmented training data or a pretrained model. The output is a grayscale image with the same spatial dimensions as the input. Pixel positions and the image scale are preserved.

VGC provides an improved input image for an existing measurement algorithm. Indentation extraction and diagonal-length measurement are performed by the downstream measurement pipeline. In this study, the downstream methods are maximum internal connected field (MICF) and flood fill (FF). Both use conventional image-processing operations and require no additional neural-network measurement stage.

## Downstream processing mechanisms

### MICF

MICF identifies the indentation through connectivity in a binary image. Its processing includes:

1. Apply the original image preprocessing, including contrast-limited adaptive histogram equalization (CLAHE) and filtering.
2. Convert the image into a binary representation using thresholding.
3. Apply morphological processing and select the largest connected region as the candidate indentation.
4. Supply the extracted region to the original geometric measurement procedure to obtain the two diagonal lengths.

Background scratches or texture connected to the candidate region can distort its extent. Missing or disconnected indentation regions can also affect the subsequent measurement.

### FF

FF identifies the indentation by growing a connected region according to gray-level similarity. Its processing includes:

1. Apply the original image preprocessing, including CLAHE and filtering.
2. Identify a candidate indentation region and select a dark seed within it.
3. Grow a connected region from the seed using the gray-level acceptance criterion of the original FF procedure.
4. Supply the resulting region to the original geometric measurement procedure to obtain the two diagonal lengths.

Local reflections and intensity variation can prevent region growth from covering the full indentation. Similar gray levels in adjacent background structures can also lead to unwanted connections.

The two methods therefore respond differently to image interference: MICF relies on the connectivity of the thresholded region, while FF relies on connected region growth under a gray-level criterion.

## Using the methods with VGC

Both downstream pipelines accept grayscale indentation images. The indentation boundary and all four vertices should be clearly resolved, and the complete indentation should be within the field of view.

The processing order is:

```text
Direct measurement
Original image
  -> Original MICF or FF pipeline, including its internal preprocessing
  -> Diagonal-length measurement
  -> Hardness calculation

VGC-assisted measurement
The same original image
  -> VGC
  -> Reconstructed grayscale image
  -> The same MICF or FF pipeline, including its internal preprocessing
  -> Diagonal-length measurement
  -> Length correction
  -> Hardness calculation
```

To incorporate VGC into an existing pipeline, replace the image supplied to that pipeline with the VGC-reconstructed image. Retain the original downstream preprocessing, extraction, image calibration, and geometric measurement procedures. In particular, the internal CLAHE operations of MICF and FF remain part of their pipelines after VGC is added.

Use the existing measurement procedure to obtain both diagonal lengths from the extracted region. Keep the test force and image calibration associated with the original image when calculating hardness.

## Comparing measurement results

Evaluate each downstream algorithm on the same images before and after adding VGC, using the same manual reference measurements. The final measurement accuracy depends jointly on VGC preprocessing and the downstream algorithm. The comparison therefore focuses on the change in measurement accuracy for a given algorithm on the same samples.

For diagonal-length evaluation, calculate the absolute percentage error of each diagonal against its corresponding manual reference length. Report the group mean as MAPE and the sample standard deviation of those errors as SD, consistent with the manuscript. Local extraction errors can affect the two pairs of opposite vertices unequally, so the MAPEs of the two diagonals need not be similar.

## Code availability

The implementation code, auxiliary tools, and experimental data will be released upon acceptance of the manuscript. Installation requirements and executable usage examples will accompany the release. This document currently describes the processing mechanisms and integration workflow.
