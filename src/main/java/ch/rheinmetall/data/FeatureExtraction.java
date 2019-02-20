package ch.rheinmetall.data;

import java.io.IOException;
import java.util.ArrayList;

import com.csvreader.CsvReader;

import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;

public class FeatureExtraction {

	DMatrix test;
	DMatrix train;
	
	public FeatureExtraction() {
		
	}

	public FeatureExtraction loadFeatures(String classFile, float trainPercent) throws IOException, XGBoostError {
		

	    int featureLength = 0;
	    ArrayList<Float> anyLabels = new ArrayList<Float>();
	    ArrayList<float[]> anyTrain = new ArrayList<float[]>();
	
		System.out.println("Reading file...");
		CsvReader myReader = new CsvReader(classFile);
		
		while(myReader.readRecord()) {
		
			String[] toks = myReader.getValues();
			anyLabels.add(new Float(toks[0]));
			
			featureLength = toks.length-1;
			float[] feats = new float[toks.length-1];
			for(int i = 0; i < 64; i++) {
				feats[i] = (new Float(toks[i+1])).floatValue();
			}
			anyTrain.add(feats);
		}
	
		int num_train = (int)(anyLabels.size()*trainPercent);
		int num_test = anyLabels.size() - num_train;
		
		float[] data = new float[num_train*featureLength];
		float[] labels = new float[num_train];
	
		
		for(int i = 0; i < num_train; i++) {
			
			labels[i] = anyLabels.get(i);
		    float[] vals = anyTrain.get(i);
			for(int k = 0; k < featureLength; k++) {	
				data[featureLength*i + k] = vals[k]; 				
			}
		}
		
		float[] datatest = new float[num_test*featureLength];
		float[] labelstest = new float[num_test];
		
		for(int i = 0; i < num_test; i++) {
			
			labelstest[i] = anyLabels.get(i+num_train);
		    float[] vals = anyTrain.get(i+num_train);
			for(int k = 0; k < featureLength; k++) {	
				datatest[featureLength*i + k] = vals[k]; 				
			}
		}
		
		System.out.println("Create testing set...");
		
		train = new DMatrix(data, num_train, featureLength);
		test  = new DMatrix(datatest, num_test, featureLength);

		train.setLabel(labels);
		test.setLabel(labelstest);
		
		//save dmatrix into binary buffer
		train.saveBinary("resources/data/model_data/features_train.buffer");
		test.saveBinary("resources/data/model_data/features_test.buffer");
		
		return this;
	}		
		
	
	
	public static void main(String[] args) throws IOException, XGBoostError {
				
		String classFile = "/home/lisztian/eclipse-workspace/0000/features/output.csv";
		float trainPercent = .8f;
		FeatureExtraction anyFeatures = new FeatureExtraction().loadFeatures(classFile, trainPercent);	
				
	}

	public DMatrix getTrainMatrix() {
		// TODO Auto-generated method stub
		return train;
	}

	public DMatrix getTestMatrix() {
		// TODO Auto-generated method stub
		return test;
	}
	
}
