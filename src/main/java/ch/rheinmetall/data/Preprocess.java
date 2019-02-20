package ch.rheinmetall.data;

import boofcv.deepboof.ClipAndReduce;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.ImageType;
import boofcv.struct.image.Planar;

public class Preprocess {

	// size of square image
	protected int imageSize;

	//  Input image adjusted to network input size
	protected Planar<GrayF32> imageRgb;

	protected ImageType<Planar<GrayF32>> imageType = ImageType.pl(3,GrayF32.class);
	
	// Resizes input image for the network
	protected ClipAndReduce<Planar<GrayF32>> massage = new ClipAndReduce<>(true, imageType);
	
	
	
	
	/**
	 * Massage the input image into a format recognized by the network
	 */
	protected  Planar<GrayF32> preprocess(Planar<GrayF32> image) {
		// Shrink the image to input size
		if( image.width == imageSize && image.height == imageSize ) {
			this.imageRgb.setTo(image);
		} else if( image.width < imageSize || image.height < imageSize ) {
			throw new IllegalArgumentException("Image width or height is too small");
		} else {
			massage.massage(image,imageRgb);
		}
		return imageRgb;
	}

	
}
