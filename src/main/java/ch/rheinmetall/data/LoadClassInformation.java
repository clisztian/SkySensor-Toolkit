package ch.rheinmetall.data;

import java.io.IOException;
import java.util.ArrayList;
import com.csvreader.CsvReader;

import ml.dmlc.xgboost4j.java.XGBoostError;

public class LoadClassInformation {

	
	private ArrayList<String> classInfo;
	
	public LoadClassInformation() {
		classInfo = new ArrayList<String>();
	}
	
	public String getClass(int i) {
		
		if(!classInfo.isEmpty() && i < classInfo.size()) {
			return classInfo.get(i);
		}
		else {
			return "Class does not exist";
		}	
	}
	
	public void loadClassInformation(String classFile) throws IOException {
		
		
		System.out.println("Reading file...");
		CsvReader myReader = new CsvReader(classFile);
		myReader.readHeaders();
		
		String[] headers = myReader.getHeaders();		
		while(myReader.readRecord()) {
			
			String myClass = myReader.get("Class");
			if(!myClass.equals("")) {
				
				String symb = myReader.get("Class Symbol");
				String symbAdd = myReader.get("Class Symbol Addon");
				String Role = myReader.get("Role");
				String Type = myReader.get("Type");
				String Indentification = myReader.get("Indentification");
				
				String all = symbAdd + " " + Type + " " + Role + 
						" - " + Indentification + ", " + myClass;
				
				classInfo.add(all);
				//System.out.println(all);
			}
			
			
		}
		
	}
	
	public static void main(String[] args) throws IOException, XGBoostError {
		
		LoadClassInformation info = new LoadClassInformation();
		info.loadClassInformation("resources/classes.csv");  
		
	}
	
	
}
