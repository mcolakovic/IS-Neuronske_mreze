package is.ispit;

import java.util.ArrayList;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.exam.NeurophExam;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.data.norm.MaxNormalizer;
import org.neuroph.util.data.norm.Normalizer;

public class MojZadatak implements NeurophExam,LearningEventListener{

    int inputCount = 30;
    int outputCount = 1;
    DataSet trainSet;
    DataSet testSet;
    double[] learningRates = {0.2,0.4,0.6};
    double learningRate;
    int[] hiddenNeurons = {10,20,30};
    int hiddenNeuron;
    int numOfTrainings = 0;
    int numOfIterations = 0;
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
        DataSet ds = DataSet.createFromFile("breast_cancer_data.csv", inputCount, outputCount, ",");
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
        learningRule.setMaxIterations(1000);
        learningRule.setMomentum(0.7);
        
        mlp.learn(trainSet);
        
        numOfTrainings++;
        numOfIterations+=learningRule.getCurrentIteration();
        
        System.out.println("Srednja vrijednost broja iteracija je: " + (double)numOfIterations/numOfTrainings);
        
        evaluate(mlp, testSet);
        
        return mlp;
    }

    @Override
    public void evaluate(MultiLayerPerceptron mlp, DataSet ds) {
        double sumError = 0, mse;
        for(DataSetRow row : ds){
            mlp.setInput(row.getInput());
            mlp.calculate();
            
            double[] actual = row.getDesiredOutput();
            double[] predicted = mlp.getOutput();
            
            sumError+=(double)Math.pow((actual[0]-predicted[0]), 2);
        }
        mse = sumError/(2*testSet.size());
        System.out.println("Srednja kvadratna greska je: " + mse);
        
        Training t = new Training(mlp, mse);
        trainings.add(t);
    }

    @Override
    public void saveBestNetwork() {
        Training minTraining = trainings.get(0);
        for(Training t : trainings)
            if(t.getMse() < minTraining.getMse())
                minTraining = t;
        minTraining.getNeuralNet().save("nn.nnet");
    }

    @Override
    public void handleLearningEvent(LearningEvent le) {
        MomentumBackpropagation bp = (MomentumBackpropagation)le.getSource();
        System.out.println("Iteration: " + bp.getCurrentIteration() + " Total network error: " + bp.getTotalNetworkError());
    }
    
}