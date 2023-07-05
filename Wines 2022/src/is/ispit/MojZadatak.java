package is.ispit;

import java.util.ArrayList;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.eval.classification.ConfusionMatrix;
import org.neuroph.exam.NeurophExam;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;
import org.neuroph.util.data.norm.MaxNormalizer;
import org.neuroph.util.data.norm.Normalizer;

public class MojZadatak implements NeurophExam,LearningEventListener{

    int inputCount = 13;
    int outputCount = 3;
    DataSet trainSet;
    DataSet testSet;
    int numOfIterations = 0;
    int numOfTrainings = 0;
    double[] learningRates = {0.2,0.4,0.6};
    ArrayList<Training> trainings = new ArrayList<>();
    
    private void run(){
        DataSet ds = loadDataSet();
        preprocessDataSet(ds);
        DataSet[] trainAndTest = trainTestSplit(ds);
        trainSet = trainAndTest[0];
        testSet = trainAndTest[1];
        MultiLayerPerceptron neuralNet = createNeuralNetwork();
        trainNeuralNetwork(neuralNet, ds);
        saveBestNetwork();
    }
    
    public static void main(String[] args) {
        new MojZadatak().run();
    }
    
    @Override
    public DataSet loadDataSet() {
        DataSet ds = DataSet.createFromFile("wines.csv", inputCount, outputCount, ",");
        return ds;
    }

    @Override
    public DataSet preprocessDataSet(DataSet ds) {
        Normalizer norm = new MaxNormalizer(ds);
        norm.normalize(ds);
        ds.shuffle();
        return ds;
    }

    @Override
    public DataSet[] trainTestSplit(DataSet ds) {
        return ds.split(0.7,0.3);
    }

    @Override
    public MultiLayerPerceptron createNeuralNetwork() {
        return new MultiLayerPerceptron(TransferFunctionType.TANH, inputCount,22,outputCount);
    }

    @Override
    public MultiLayerPerceptron trainNeuralNetwork(MultiLayerPerceptron mlp, DataSet ds) {
        for(double lr : learningRates){
            BackPropagation learningRule = (BackPropagation)mlp.getLearningRule();
            learningRule.addListener(this);
            learningRule.setMaxIterations(1000);
            learningRule.setLearningRate(lr);
            learningRule.setMaxError(0.02);
            
            mlp.learn(trainSet);
            
            numOfTrainings++;
            numOfIterations+=learningRule.getCurrentIteration();
            
            evaluate(mlp, testSet);
        }
        System.out.println("Srednja vrijednost broja iteracija je: " + (double)numOfIterations/numOfTrainings);
        return mlp;
    }

    @Override
    public void evaluate(MultiLayerPerceptron mlp, DataSet ds) {
        String[] labels = {"c1","c2","c3"};
        ConfusionMatrix cm = new ConfusionMatrix(labels);
        double accuracy = 0;
        
        for(DataSetRow row : ds){
            mlp.setInput(row.getInput());
            mlp.calculate();
            
            int actual = getMaxIndex(row.getDesiredOutput());
            int predicted = getMaxIndex(mlp.getOutput());
            
            cm.incrementElement(actual, predicted);
        }
        
        for (int i = 0; i < outputCount; i++) {
            accuracy+=(double)(cm.getTruePositive(i)+cm.getTrueNegative(i))/cm.getTotal();
        }
        
        System.out.println(cm.toString());
        accuracy = accuracy/outputCount;
        System.out.println("Moj accuracy je: " + accuracy);
        
        Training t = new Training(mlp, accuracy);
        trainings.add(t);
    }

    @Override
    public void saveBestNetwork() {
        Training maxTraining = trainings.get(0);
        for(Training t : trainings)
            if(t.getAccuracy() > maxTraining.getAccuracy())
                maxTraining = t;
        maxTraining.getNeuralNet().save("nn.nnet");
    }

    @Override
    public void handleLearningEvent(LearningEvent le) {
        BackPropagation bp = (BackPropagation)le.getSource();
        System.out.println("Iteration: " + bp.getCurrentIteration() + " Total network error: " + bp.getTotalNetworkError());
    }

    private int getMaxIndex(double[] output) {
        int max = 0;
        for (int i = 0; i < output.length; i++)
            if(output[i] > output[max])
                max = i;
        return max;
    }
    
}