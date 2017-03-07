package de.l3s.souza.svm;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map.Entry;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.functions.LibSVM;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.stopwords.StopwordsHandler;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

 
public class WekaSVM {
	
	private static LibSVM libSVM;
	private static Instances dataFiltered;
	private static Instances data;
	private static StringToWordVector filter;
	
	public WekaSVM ()
	{
		filter = new StringToWordVector();
	}
	
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;
 
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}
 
		return inputReader;
	}
 
	public static Evaluation classify(Classifier model,
			Instances trainingSet, Instances testingSet) throws Exception {
		Evaluation evaluation = new Evaluation(trainingSet);
 
		model.buildClassifier(trainingSet);
		evaluation.evaluateModel(model, testingSet);
 
		return evaluation;
	}
 
	public static double calculateAccuracy(ArrayList<Prediction> predictions) {
		double correct = 0;
 
		for (int i = 0; i < predictions.size(); i++) {
			NominalPrediction np = (NominalPrediction) predictions.get(i);
			if (np.predicted() == np.actual()) {
				correct++;
			}
		}
 
		return 100 * correct / predictions.size();
	}
 
	public static Instances[][] crossValidationSplit(Instances data, int numberOfFolds) {
		Instances[][] split = new Instances[2][numberOfFolds];
 
		for (int i = 0; i < numberOfFolds; i++) {
			split[0][i] = data.trainCV(numberOfFolds, i);
			split[1][i] = data.testCV(numberOfFolds, i);
		}
 
		return split;
	}
 
	public static class GermanStopwordsHandler implements StopwordsHandler {

	    private HashSet<String> stopwords;

	    public GermanStopwordsHandler() {
	    	
	    	stopwords = new HashSet<String>();
	    	try 
	    	  {
	    	   File fl = new File("./stopwords.txt");
	    	   BufferedReader br = new BufferedReader(new FileReader(fl)) ;
	    	   String str;
	    	   while ((str=br.readLine())!=null)
	    	   {
	    		   stopwords.add(str);
	    	   }
	    	   br.close();
	    	   
	    	  }
	    	  catch (IOException  e)
	    	  { e.printStackTrace(); }
	    }

	    
	    public boolean isStopword(String word) {
	        return stopwords.contains(word); 
	    }

	}
	public void runSVM() throws Exception {
	//public static void main(String[] args) throws Exception {
		BufferedReader datafile = readDataFile("original_data.arff");
 		data = new Instances(datafile);
		data.setClassIndex(0);
		filter = new StringToWordVector();
		filter.setAttributeIndices("first-last");
		filter.setInputFormat(data);
		
		filter.setDoNotOperateOnPerClassBasis(false);
		filter.setOutputWordCounts(true);
		filter.setWordsToKeep(1000);
		filter.setIDFTransform(true);
		filter.setTFTransform(true);
		filter.setMinTermFreq(3);
		filter.setNormalizeDocLength(new SelectedTag(StringToWordVector.FILTER_NORMALIZE_ALL,StringToWordVector.TAGS_FILTER));
		filter.setDoNotOperateOnPerClassBasis(false);
		filter.setPeriodicPruning(-1);
		
		filter.setStopwordsHandler(new GermanStopwordsHandler());
		//filter.setS
	//	String[] options = filter.getOptions();
	/*	for(int i=0;i<options.length;i++) {
		if (options[i].length() > 0)
		System.out.println(options[i]);
		}
*/
		
	try {	
		dataFiltered = Filter.useFilter(data, filter);
	} catch (Exception e){
		
		e.printStackTrace();
		
	}
		
		if (dataFiltered != null)
			System.out.println("\n\n=====> Filtered data:<===\n\n" + dataFiltered.toString());
		else
			System.out.println ("empty data");
		dataFiltered.setClassIndex(0);
 
		// Do 10-split cross validation
		Instances[][] split = crossValidationSplit(dataFiltered, 2);
 
		// Separate split into training and testing arrays
		Instances[] trainingSplits = split[0];
		Instances[] testingSplits = split[1];
 
		// Use a set of classifiers
		/*
		Classifier[] models = { 
				
				new LibSVM()
		};
		 */
		libSVM = new LibSVM();
		
		// Run for each model
//		for (int j = 0; j < models.length; j++) {
 
			// Collect every group of predictions for current model in a FastVector
			ArrayList<Prediction> predictions = new ArrayList<Prediction>();
 
			// For each training-testing split pair, train and test the classifier
			for (int i = 0; i < trainingSplits.length; i++) {
				
				Evaluation validation = classify(libSVM, trainingSplits[i], testingSplits[i]);
 
				predictions.addAll(validation.predictions());
 
				// Uncomment to see the summary for each training-testing pair.
//				System.out.println(libSVM.toString());
			}
 
			// Calculate overall accuracy of current classifier on all splits
			double accuracy = calculateAccuracy(predictions);
 
			// Print current classifier's name and accuracy in a complicated,
			// but nice-looking way.
			System.out.println("Accuracy of " + libSVM.getClass().getSimpleName() + ": "
					+ String.format("%.2f%%", accuracy)
					+ "\n---------------------------------");

	}

	public HashMap<String,Double> classifyInstance (HashMap<String,String> documents) throws Exception
	{
		HashMap<String,Double> classifiedInstances = new HashMap<String,Double>();
		
		ArrayList<Attribute> attributes = new ArrayList<Attribute>(2);
		ArrayList<String> classVal = new ArrayList<String>();
		classVal.add("positive");
	    classVal.add("negative");
		double result = 0;
		
	    attributes.add(new Attribute("article",(ArrayList<String>)null));
	    attributes.add(new Attribute("relevance",classVal));
	    
		Instances instances = new Instances("Test",attributes,0);
		instances.setClassIndex(0);
		
		Instance instance = null;
		
		 for(Entry<String, String> entry : documents.entrySet())
         {
			 	String key = entry.getKey();
        	    instance = makeInstance(key,instances);
        	    data.add(instance);
        	    data.instance(data.numInstances()-1).setValue(0,key);
        	    dataFiltered = Filter.useFilter(data, filter);
        	    dataFiltered.setClassIndex(0);
        	    result = libSVM.classifyInstance(dataFiltered.lastInstance());
        	    classifiedInstances.put(key, result);
         }
//		 System.out.println (dataFiltered.toString());
		// System.out.printf("%s %f ",dataFiltered.lastInstance(),result);
		 
//		instances = buildArff(documents,"Test");
//		System.out.println (instances.toString());
//		libSVM.classifyInstance(instances);
		 
		 return classifiedInstances;
	}
	public static Instances buildArff(HashMap<String,String> documents, String relationName) throws Exception
	  {
			 ArrayList<Attribute> atts = new ArrayList<Attribute>(2);
			 ArrayList<String> classVal = new ArrayList<String>();
			 classVal.add("positive");
		     classVal.add("negative");
			 atts.add(new Attribute("article",(ArrayList<String>)null));
		     atts.add(new Attribute("relevance",classVal));
		        
	         // 2. create Instances object
	         Instances test = new Instances(relationName, atts, 0);
	         
	         System.out.println("Before adding any instance");
		        System.out.println("--------------------------");
		        System.out.println(test);
		        System.out.println("--------------------------");
	         for(Entry<String, String> entry : documents.entrySet())
	         {
	        	    String key = entry.getKey();
	        	    String value = entry.getValue();
	        	    
	        	    double vals[] = new double[test.numAttributes()];
	        	    vals[0]=test.attribute(0).addStringValue(key);
	        	    vals[1]=test.attribute(1).addStringValue(value);
	        	    test.add(new DenseInstance(1.0, vals));
	        	    
	        }
	         System.out.println("After adding a instance");
	         System.out.println("--------------------------");
	         System.out.println(test);
	         System.out.println("--------------------------");
	         return(test);
	  }
	
	  	private static Instance makeInstance(String text, Instances data) {

		    // Create instance of length two.
		    Instance instance = new DenseInstance(2);

		    // Set value for message attribute
		    Attribute messageAtt = data.attribute("article");
		    instance.setValue(messageAtt, messageAtt.addStringValue(text));
		    
		    // Give instance access to attribute information from the dataset.
		    instance.setDataset(data);
		    
		    return instance;
		  }
	/*
	public static void classifyUnlabeledInstances () throws Exception
	{
		Instances unlabeled = new Instances(
                new BufferedReader(
                  new FileReader("unlabeled.arff")));
		unlabeled.setClassIndex(unlabeled.numAttributes() - 1);
		Instances labeled = new Instances(unlabeled);
		
		// label instances
		 for (int i = 0; i < unlabeled.numInstances(); i++) {
			 
		   double clsLabel = libSVM.classifyInstance(unlabeled.instance(i));
		   labeled.instance(i).setClassValue(clsLabel);
		 }
		
		// save labeled data
		 BufferedWriter writer = new BufferedWriter(
		                           new FileWriter("labeled.arff"));
		 writer.write(labeled.toString());
		 writer.newLine();
		 writer.flush();
		 writer.close();
		
	}
	*/
}