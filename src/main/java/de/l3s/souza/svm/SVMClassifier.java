package de.l3s.souza.svm;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Enumeration;
import java.util.Vector;

import libsvm.*;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.SelectedTag;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class SVMClassifier extends AbstractClassifier implements OptionHandler, WeightedInstancesHandler {
	static final long serialVersionUID = 2;
    svm_parameter param;
    svm_problem prob;
    svm_model model;
    String[] options = new String[18];
    boolean if_debug;
    int n_x;
    svm_node[] x;

    int cross_validation = 0;
    int predict_probability = 0;
    int nr_fold = 1;

    // set by parse_command_line
    String model_file_name;		

    public SVMClassifier() {
	param = new svm_parameter();
	prob = new svm_problem();
	InitParam();
    }

    private void InitParam() {
	// default values
    	StringToWordVector filter = new StringToWordVector();
   
    
    param.svm_type = svm_parameter.NU_SVC;
	param.kernel_type = svm_parameter.RBF;
	param.degree = SVMConstants.DEFAULT_DEGREE;
	param.gamma = SVMConstants.DEFAULT_GAMMA;	// 1/k
	param.coef0 = SVMConstants.DEFAULT_COEF0;
	param.nu = SVMConstants.DEFAULT_NU;
	param.cache_size = SVMConstants.DEFAULT_CACHE_SIZE;
	param.C = SVMConstants.DEFAULT_C;
	param.eps = SVMConstants.DEFAULT_EPS;
	param.p = SVMConstants.DEFAULT_P;
	param.shrinking = SVMConstants.DEFAULT_SHRINKING;
	param.probability = SVMConstants.DEFAULT_PROB;
	param.nr_weight = SVMConstants.DEFAULT_NR_WEIGHT;
	param.weight_label = new int[0];
	param.weight = new double[0];
	
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
    public void buildClassifier(Instances data) throws Exception {
    	
    	
	Exception e = new Exception("BuildClassifier failed");
        if (!ReadData(data))
            throw e;
	System.out.println("parameter: degree: " + param.degree + 
			   "  gamma: " + param.gamma + "  coef0: " + param.coef0 + 
			   "  nu: " + param.nu + "  cache_size: " + param.cache_size + 
			   "  C: " + param.C + "  eps: " + param.eps + 
			   "  p: " + param.p + "  shrinking: " + param.shrinking + 
			   "  probability: " + param.probability + "  nr_weight: " + param.nr_weight);
        Train();
    }

    public double classifyInstance(Instance data) throws Exception {
	Exception e = new Exception("ClassifyInstance failed");
        if (!ReadData(data))
            throw e;
	
	double[] temp_double = data.toDoubleArray();
	
	System.out.print("SVM::classifyInstance: data: ");
	for(int i = 0; i < data.numAttributes(); i++)
	    System.out.print(temp_double[i] + " ");
        double temp = Test();
	System.out.println("class: " + temp);
	return temp;

    }
    
    /*
    private void do_cross_validation() {
	int i;
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double[] target = new double[prob.l];
	
	svm.svm_cross_validation(prob,param,nr_fold,target);
	if(param.svm_type == svm_parameter.EPSILON_SVR ||
	   param.svm_type == svm_parameter.NU_SVR) {
	    for(i=0;i<prob.l;i++) {
		double y = prob.y[i];
		double v = target[i];
		total_error += (v-y)*(v-y);
		sumv += v;
		sumy += y;
		sumvv += v*v;
		sumyy += y*y;
		sumvy += v*y;
	    }
	    System.out.print("Cross Validation Mean squared error = "+total_error/prob.l+"\n");
	    System.out.print("Cross Validation Squared correlation coefficient = "+
			     ((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
			     ((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))+"\n"
			     );
	}
	else {
	    for(i=0;i<prob.l;i++)
		if(target[i] == prob.y[i])
		    ++total_correct;
	}
	System.out.print("Cross Validation Accuracy = "+100.0*total_correct/prob.l+"%\n");
	}
    */


    public void Train() throws IOException {
    	
	model = svm.svm_train(prob,param);
	System.out.println("SVM::Train done\n");
    }

    public double Test() {
	int correct = 0;
	int total = 0;
	double error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	
	int svm_type = svm.svm_get_svm_type(model);
	int nr_class = svm.svm_get_nr_class(model);
	int[] labels = new int[nr_class];
	double[] prob_estimates = null;


	if(predict_probability == 1) {
	    if(svm_type == svm_parameter.EPSILON_SVR ||
	       svm_type == svm_parameter.NU_SVR) {
		System.out.print("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma="+svm.svm_get_svr_probability(model)+"\n");
	    }
	    else {
		svm.svm_get_labels(model,labels);
		prob_estimates = new double[nr_class];
	    }
	}
	double v;
	if (predict_probability==1 && (svm_type==svm_parameter.C_SVC || svm_type==svm_parameter.NU_SVC)) {
	    v = svm.svm_predict_probability(model,x,prob_estimates);
	}
	else {
	    v = svm.svm_predict(model,x);
	}
	return v;
    }

    private boolean ReadData(Instance data) {
	if(!CheckData(data))
	    return false;
	int n_attrib = data.numAttributes();
	double[] temp_double = data.toDoubleArray();
	double class_id = data.classIndex();

	n_x = n_attrib - 1;  
	x = new svm_node[n_x];
	for(int j=0, current = 0;j<n_attrib;j++) {
	    if(j == class_id) continue;
	    x[current] = new svm_node();
	    x[current].index = current;
	    x[current].value = temp_double[j]; 
	    current++;
	}

	return true;
    }

    private boolean ReadData(Instances data) {
	Vector vy = new Vector();
	Vector vx = new Vector();
	int n_data = data.numInstances();
	int n_attrib = data.numAttributes();

	int max_index = 0;

	double[] n_attrib_values = new double[n_attrib];
	double[] min_attrib_values = new double[n_attrib];
	double[] max_attrib_values = new double[n_attrib];
	double[] temp_double = null;
	int class_id = data.classIndex();

	n_x = n_attrib - 1; //y in the last
	//System.out.println("n_x: " + n_x);
	
	for(int i = 0; i < n_attrib; i++) {
	    n_attrib_values[i] = data.instance(0).attribute(i).numValues();
	    min_attrib_values[i] = data.instance(0).value(i);
	    max_attrib_values[i] = data.instance(0).value(i);
	    for(int j = 1; j < n_data; j++) {
		min_attrib_values[i] = 
		    min_attrib_values[i] < data.instance(j).value(i) ? 
		    min_attrib_values[i] : data.instance(j).value(i);
		max_attrib_values[i] = 
		    max_attrib_values[i] > data.instance(j).value(i) ? 
		    max_attrib_values[i] : data.instance(j).value(i);
	    }
	}

        for (int i = 0; i < n_data; i++) {
            if (!CheckData(data.instance(i)))
                return false;
	    temp_double = data.instance(i).toDoubleArray();

	    vy.addElement(data.instance(i).classValue()); 
	    x = new svm_node[n_x]; 
	    for(int j = 0, current = 0; j < n_attrib; j++) {
		if(j == class_id) continue;
		x[current] = new svm_node();
		x[current].index = current;
		if(n_attrib_values[j] != 0) 
		    x[current].value = 
			(temp_double[j] - min_attrib_values[j]) / 
			(max_attrib_values[j] - min_attrib_values[j]);
		else
		    x[current].value = temp_double[j];
		current++;
	    }

	    if(n_x > 0) max_index = Math.max(max_index, x[n_x - 1].index);
	    vx.addElement(x);

	    /*
	    System.out.print("input: \n");
	    for(int j = 0; j < x.length; j++)
		System.out.print(x[j].value + " ");
	    System.out.print(data.instance(i).classValue() + "\n");
	    */
	}

	prob.l = n_data;
	prob.x = new svm_node[prob.l][];
	for(int i = 0; i < prob.l; i++)
	    prob.x[i] = (svm_node[])vx.elementAt(i);

	prob.y = new double[prob.l];
	for(int i = 0; i < prob.l; i++) 
	    prob.y[i] = atof(vy.elementAt(i).toString());
	
	if(param.gamma == 0)
	    param.gamma = 1.0/n_x;

        return true;
    }

    private boolean CheckData(Instance data) {
        return true;
    }
    
    private double atof(String s) {
	//System.out.println(s);
	return Double.parseDouble(s);
    }

    private static int atoi(String s) {
	return Integer.parseInt(s);
    }


    public String globalInfo() {
	return "SVM in libsvm";
    }

    public Enumeration listOptions() {
	Vector newVector = new Vector(3);
	newVector.addElement(new Option("\tTurn on debugging output.",
					"D", 0, "-D"));
	newVector.addElement(new Option("\tSet the ridge in the log-likelihood.",
					"R", 1, "-R <ridge>"));
	newVector.addElement(new Option("\tSet the maximum number of iterations"+
					" (default -1, until convergence).",
					"M", 1, "-M <number>"));
	return newVector.elements();
    }
    
    public void setOptions(String[] options) throws Exception {
	setDebug(Utils.getFlag('D', options));
	String temp;
	
//	temp = Utils.getOption('t', options);
	temp = Utils.getOption('d', options);
	temp = Utils.getOption('g', options);
	temp = Utils.getOption('r', options);
	temp = Utils.getOption('c', options);
	temp = Utils.getOption('n', options);
	temp = Utils.getOption('p', options);
	temp = Utils.getOption('m', options);
	temp = Utils.getOption('e', options);
	temp = Utils.getOption('h', options);
	temp = Utils.getOption('b', options);
	temp = Utils.getOption('e', options);
    }
    
    public String [] getOptions() {
	String [] options = new String [SVMConstants.N_OPTION_LINE];
	int current = 0;
	
	if (getDebug()) 
	    options[current++] = "-D";
	options[current++] = "-s";
	options[current++] = "" + getSVMType();	
//	options[current++] = "-t";
//	options[current++] = "" + getKernelType();	
	options[current++] = "-d";
	options[current++] = "" + getDegree();
	options[current++] = "-g";
	options[current++] = "" + getGamma();
	options[current++] = "-r";
	options[current++] = "" + getCoef0();
	options[current++] = "-c";
	options[current++] = "" + getC();
	options[current++] = "-n";
	options[current++] = "" + getNu();
	options[current++] = "-p";
	options[current++] = "" + getP();
	options[current++] = "-m";
	options[current++] = "" + getCacheSize();
	options[current++] = "-e";
	options[current++] = "" + getEps();
	options[current++] = "-h";
	options[current++] = "" + getShrinking();
	options[current++] = "-b";
	options[current++] = "" + getProbability();
	while (current < options.length) 
	    options[current++] = "";
	return options;
    }
    
    public String debugTipText() {
	return "Output debug information to the console.";
    }
    public void setDebug(boolean temp) {
	if_debug = temp;
    }
    public boolean getDebug() {
	return if_debug;
    }      

    public String SVMTypeTipText() {
	return "0 = linear: u'*v \n1 = polynomial: (gamma * u'*v + coef0)^degree\n2 = radial basis function: exp(-gamma*|u-v|^2)\n3 = sigmoid: tanh(gamma*u'*v + coef0)";
    }
    public void setSVMType(int temp) {
	param.svm_type = temp;
    }
    public int getSVMType() {
	return param.svm_type;
    }
    
    public String KernelTypeTipText() {
	return "0 = linear: u'*v \n1 = polynomial: (gamma * u'*v + coef0)^degree\n2 = radial basis function: exp(-gamma*|u-v|^2)\n3 = sigmoid: tanh(gamma*u'*v + coef0)";
    }
    public void setKernelType(int temp) {
	param.kernel_type = temp;
    }
    public int getKernelType() {
	return param.kernel_type;
    }
    
    public String DegreeTipText() {
	return "set degree in kernel function (default 3)";
    }
    public void setDegree(int temp) {
	param.degree = temp;
    }    
    public double getDegree() {
	return param.degree;
    }

    public String GammaTipText() {
	return "set gamma in kernel function (default 3)";
    }
    public void setGamma(double temp) {
	param.gamma = temp;
    }    
    public double getGamma() {
	return param.gamma;
    }

    public String Coef0TipText() {
	return "set coef0 in kernel function (default 3)";
    }
    public void setCoef0(double temp) {
	param.coef0 = temp;
    }    
    public double getCoef0() {
	return param.coef0;
    }
    
    public String CTipText() {
	return "set cost in kernel function (default 3)";
    }
    public void setC(double temp) {
	param.C = temp;
    }    
    public double getC() {
	return param.C;
    }

    public String NuTipText() {
	return "set nu in kernel function (default 3)";
    }
    public void setNu(double temp) {
	param.nu = temp;
    }    
    public double getNu() {
	return param.nu;
    }

    public String PTipText() {
	return "set p in kernel function (default 3)";
    }
    public void setP(double temp) {
	param.p = temp;
    }    
    public double getP() {
	return param.p;
    }

    public String CacheSizeTipText() {
	return "set cacheSize in kernel function (default 3)";
    }
    public void setCacheSize(double temp) {
	param.cache_size = temp;
    }    
    public double getCacheSize() {
	return param.cache_size;
    }

    public String EpsTipText() {
	return "set eps in kernel function (default 3)";
    }
    public void setEps(double temp) {
	param.eps = temp;
    }    
    public double getEps() {
	return param.eps;
    }

    public String ShrinkingTipText() {
	return "set shrinking in kernel function (default 3)";
    }
    public void setShrinking(int temp) {
	param.shrinking = temp;
    }    
    public int getShrinking() {
	return param.shrinking;
    }

    public String ProbabilityTipText() {
	return "set probability in kernel function (default 3)";
    }
    public void setProbability(int temp) {
	param.probability = temp;
    }    
    public int getProbability() {
	return param.probability;
    }

    public static void main(String[] args) {
        try {
//            System.out.println(Evaluation.evaluateModel(new SVMClassifier(), args));
        	SVMClassifier svm = new SVMClassifier ();
        	BufferedReader datafile = readDataFile("original_data.arff");
        	 
    		StringToWordVector filter = new StringToWordVector();
    		Instances data = new Instances(datafile);
    		filter.setInputFormat(data);
    		filter.setAttributeIndices("first-last");
    		
    		filter.setDoNotOperateOnPerClassBasis(false);
    		filter.setOutputWordCounts(true);
    		filter.setWordsToKeep(1000);
    		filter.setIDFTransform(true);
    		filter.setTFTransform(true);
    		filter.setMinTermFreq(3);
    		filter.setNormalizeDocLength(new SelectedTag(StringToWordVector.FILTER_NORMALIZE_ALL,StringToWordVector.TAGS_FILTER));
    		filter.setDoNotOperateOnPerClassBasis(false);
    		filter.setPeriodicPruning(-1);
    		
    		String[] options = filter.getOptions();
//    		for(int i=0;i<options.length;i++) {
//    		if (options[i].length() > 0)
//    		System.out.println(options[i]);
//    		}

    		Instances dataFiltered = Filter.useFilter(data, filter);
    		
    		dataFiltered.setClassIndex(0);
        	svm.buildClassifier(dataFiltered);
        	String options2 [] = new String [1];
        	options2[0] = "-t";
        	options2[1] = "original_data.arff";
        	AbstractClassifier.runClassifier(svm, options2);
        	
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

}