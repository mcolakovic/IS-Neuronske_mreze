package is.ispit;

import java.util.ArrayList;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.eval.classification.ConfusionMatrix;
import org.neuroph.exam.NeurophExam;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.data.norm.MaxNormalizer;
import org.neuroph.util.data.norm.Normalizer;

public class MojZadatak implements NeurophExam,LearningEventListener{

    int inputCount = 9;
    int outputCount = 7;
    DataSet trainSet;
    DataSet testSet;
    double[] learningRates = {0.2, 0.4, 0.6};
    double learningRate;
    int[] hiddenNeurons = {10, 20, 30};
    int hiddenNeuron;
    int numOfIterations = 0;
    int numOfTrainings = 0;
    ArrayList<Training> trainings = new ArrayList<>();
            
    private void run(){
        DataSet ds = loadDataSet();
        preprocessDataSet(ds);
        DataSet[] trainAndTest = trainTestSplit(ds);
        trainSet = trainAndTest[0];
        testSet = trainAndTest[1];
        
        for(double lr : learningRates){
            learningRate = lr;
            for(int hn : hiddenNeurons){
                hiddenNeuron = hn;
                MultiLayerPerceptron neuralNet = createNeuralNetwork();
                trainNeuralNetwork(neuralNet, ds);
            }
        }
        
        saveBestNetwork();
    }
    
    public static void main(String[] args) {
        new MojZadatak().run();
    }
    
    @Override
    public DataSet loadDataSet() {
        DataSet ds = DataSet.createFromFile("glass.csv", inputCount, outputCount, ",");
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
        return ds.split(0.65,0.35);
    }

    @Override
    public MultiLayerPerceptron createNeuralNetwork() {
        return new MultiLayerPerceptron(inputCount,hiddenNeuron,outputCount);
    }

    @Override
    public MultiLayerPerceptron trainNeuralNetwork(MultiLayerPerceptron mlp, DataSet ds) {
        MomentumBackpropagation learningRule = (MomentumBackpropagation)mlp.getLearningRule();
        learningRule.addListener(this);
        learningRule.setLearningRate(learningRate);
        learningRule.setMomentum(0.6);
        learningRule.setMaxIterations(1000);
        
        mlp.learn(trainSet);
        
        numOfTrainings++;
        numOfIterations+=learningRule.getCurrentIteration();
        
        evaluate(mlp, testSet);
        
        return mlp;
    }

    @Override
    public void evaluate(MultiLayerPerceptron mlp, DataSet ds) {
        String[] labels = {"c1","c2","c3","c4","c5","c6","c7"};
        ConfusionMatrix cm = new ConfusionMatrix(labels);
        double accuraccy = 0;
        
        for(DataSetRow row : ds){
            mlp.setInput(row.getInput());
            mlp.calculate();
            
            int actual = getMaxIndex(row.getDesiredOutput());
            int predicted = getMaxIndex(mlp.getOutput());
            
            cm.incrementElement(actual, predicted);
        }
        
        for (int i = 0; i < outputCount; i++)
            accuraccy += (double)(cm.getTruePositive(i)+cm.getTrueNegative(i))/cm.getTotal();
        
        System.out.println(cm.toString());
        accuraccy = accuraccy/outputCount;
        System.out.println("Moj accuraccy je: " + accuraccy);
        
        Training t = new Training(mlp, accuraccy);
        trainings.add(t);
    }

    @Override
    public void saveBestNetwork() {
        Training maxTraining = trainings.get(0);
        for(Training t : trainings)
            if(t.getAccuraccy() > maxTraining.getAccuraccy())
                maxTraining = t;
        maxTraining.getNeuralNet().save("nn.nnet");
    }

    @Override
    public void handleLearningEvent(LearningEvent le) {
        MomentumBackpropagation bp = (MomentumBackpropagation)le.getSource();
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