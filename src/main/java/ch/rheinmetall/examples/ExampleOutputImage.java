package ch.rheinmetall.examples;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import boofcv.gui.ImageClassificationPanel;
import boofcv.gui.ListDisplayPanel;
import boofcv.gui.image.ShowImages;
import boofcv.io.UtilIO;
import boofcv.io.image.ConvertBufferedImage;
import boofcv.io.image.UtilImageIO;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.Planar;
import ch.rheinmetall.data.Resize;


public class ExampleOutputImage {

	public static void main(String[] args) throws IOException {
		
		String imagePath = "resources/data/0000/classification/gradient/test/";
		List<File> images = Arrays.asList(UtilIO.findMatches(new File(imagePath),"\\w*.jpg"));
		
		Collections.sort(images);
		ImageClassificationPanel gui = new ImageClassificationPanel();
		ShowImages.showWindow(gui, "Image Classifier", true);
		
		File f = images.get(images.size()-1);
		
		BufferedImage buffered = UtilImageIO.loadImage(f.getPath());
		if( buffered == null)
			throw new RuntimeException("Couldn't find input image");

		
		System.out.println(buffered.getType());
		
		DataBufferByte dataBuffer = ((DataBufferByte)buffered.getRaster().getDataBuffer());
		int size = dataBuffer.getSize();
		float[] myImage = new float[size];
		for(int i = 0; i < size; i++) {
			myImage[i] = dataBuffer.getElemFloat(i)/255f;
		}
		


		ListDisplayPanel panel = new ListDisplayPanel();
		
		panel.addImage(buffered,"Original");
		panel.addImage(UtilImageIO.loadImage(images.get(images.size()-40).getPath()),"Orignal 2");
				
		
		BufferedImage out = Resize.scale(buffered, 64, 64);	
	
		panel.addImage(out,"Original");
		
		ShowImages.showWindow(panel,"Image Examples",true);
	}
	
}
