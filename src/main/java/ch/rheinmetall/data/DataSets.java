package ch.rheinmetall.data;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import boofcv.io.image.UtilImageIO;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;

public class DataSets {

	
	private String imagePath = "resources/data/0000/trainGrad.txt";
	private DMatrix train;
	private DMatrix test;
	private float testFraction = .1f;
	private int num_train;
	private int num_test;
	private int imageWidth = 64;
	private int imageHeight = 64;
	private ArrayList<String> imageFiles;

	
	public DataSets(float testFraction, int imageWidth, int imageHeight) {
		
		this.testFraction = testFraction;
		this.imageHeight = imageHeight;
		this.imageWidth = imageWidth;
	}
	
	
	public void loadImages() throws IOException {
		
		//classification/gradient/train/9IGVegaCore_000323747.jpg 9
		
		String line;
	    FileInputStream in = new FileInputStream( new File(imagePath));
	    BufferedReader reader = new BufferedReader(new InputStreamReader(in));
	    imageFiles = new ArrayList<String>();
	    
	    while ((line = reader.readLine()) != null) {	    	
	    	imageFiles.add(line.trim());	
	    }
	    reader.close();
	    in.close();  
	    
	    Collections.shuffle(imageFiles, new Random(5)); 
	    		
	}
	
	
	public void createDataSets() throws XGBoostError {
		
		String base = "resources/data/0000/";
		num_train = (int) ((int)imageFiles.size()*(1.0 - testFraction));
		num_test = imageFiles.size() - num_train; 
		
		float[] data = new float[num_train*imageWidth*imageHeight];
		float[] labels = new float[num_train];
		
		System.out.println("Create training set...");
		for(int i = 0; i < num_train; i++) {

			 String[] imagePathLabel = imageFiles.get(i).split("[ ]+");
			 BufferedImage out = Resize.scale(UtilImageIO.loadImage(base+imagePathLabel[0]), imageWidth, imageHeight);
			 DataBufferByte dataBuffer = ((DataBufferByte)out.getRaster().getDataBuffer());
			 
			 int size = dataBuffer.getSize();
			 for(int k = 0; k < size; k++) {
				 data[i*size + k] = dataBuffer.getElemFloat(k)/255f;
			 }
			 labels[i] = Float.valueOf(imagePathLabel[1]);
		}
		
		float[] datatest = new float[num_test*imageWidth*imageHeight];
		float[] labelstest = new float[num_test];
		
		System.out.println("Create testing set...");
		for(int i = 0; i < num_test; i++) {

			 String[] imagePathLabel = imageFiles.get(num_train + i).split("[ ]+");
			 BufferedImage out = Resize.scale(UtilImageIO.loadImage(base+imagePathLabel[0]), imageWidth, imageHeight);
			 DataBufferByte dataBuffer = ((DataBufferByte)out.getRaster().getDataBuffer());
			 
			 int size = dataBuffer.getSize();
			 for(int k = 0; k < size; k++) {
				 datatest[i*size + k] = dataBuffer.getElemFloat(k)/255f;
			 }
			 labelstest[i] = Float.valueOf(imagePathLabel[1]);
		}
		
		
		train = new DMatrix(data, num_train, imageWidth*imageHeight);
		test  = new DMatrix(datatest, num_test, imageWidth*imageHeight);

		train.setLabel(labels);
		test.setLabel(labelstest);
		
		//save dmatrix into binary buffer
		train.saveBinary("resources/data/model_data/train.buffer");
		test.saveBinary("resources/data/model_data/test.buffer");
		
	}
	
	public void createDataSetsFromBinary(String trainFile, String testFile) throws XGBoostError {
		
		train = new DMatrix(trainFile);
		test = new DMatrix(testFile);
	}
	
	
	public String getImagePath() {
		return imagePath;
	}
	public void setImagePath(String imagePath) {
		this.imagePath = imagePath;
	}
	
	public float getTestFraction() {
		return testFraction;
	}
	public void setTestFraction(float testFraction) {
		this.testFraction = testFraction;
	}
	
	public DMatrix getTestMatrix() {
		return test;
	}
	
	public DMatrix getTrainMatrix() {
		return train;
	}
	public static void main(String[] args) throws IOException, XGBoostError {
		
		int imageHeight = 64;
		int imageWidth = 64;
		float percent = .1f;
		
		DataSets anyData = new DataSets(percent, imageWidth, imageHeight);
		
		anyData.loadImages();
		anyData.createDataSets();
				
		float[] myLabels = anyData.getTestMatrix().getLabel();
		
		for(int i = 0; i < myLabels.length; i++) {
			System.out.println(myLabels[i]);
		}
 	}
	
	
	public int getImageWidth() {
		return imageWidth;
	}
	
	public int getImageHeight() {
		return imageHeight;
	}
}
