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
    
    int inputCount = 8;
    int outputCount = 1;
    DataSet trainSet;
    DataSet testSet;
    double[] learningRates = {0.2,0.4,0.6};
    int numOfIterations = 0;
    int numOfTrainings = 0;
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
        DataSet ds = DataSet.createFromFile("diabetes_data.csv", inputCount, outputCount, ",");
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
        return ds.split(0.6,0.4);
    }

    @Override
    public MultiLayerPerceptron createNeuralNetwork() {
        return new MultiLayerPerceptron(inputCount,20,16,outputCount);
    }

    @Override
    public MultiLayerPerceptron trainNeuralNetwork(MultiLayerPerceptron mlp, DataSet ds) {
        for(double lr : learningRates){
            MomentumBackpropagation learningRule = (MomentumBackpropagation)mlp.getLearningRule();
            learningRule.addListener(this);
            learningRule.setLearningRate(lr);
            learningRule.setMaxError(0.07);
            learningRule.setMomentum(0.5);
            learningRule.setMaxIterations(1000);
            
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
        String[] labels = {"c1","c2"};
        ConfusionMatrix cm = new ConfusionMatrix(labels);
        double accuracy = 0;
        
        for(DataSetRow row : ds){
            mlp.setInput(row.getInput());
            mlp.calculate();
            
            int actual = (int)Math.round(row.getDesiredOutput()[0]);
            int predicted = (int)Math.round(mlp.getOutput()[0]);
            
            cm.incrementElement(actual, predicted);
        }
        
        accuracy += (double)(cm.getTruePositive(0)+cm.getTrueNegative(0))/cm.getTotal();
        
        System.out.println(cm.toString());
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
        MomentumBackpropagation bp = (MomentumBackpropagation)le.getSource();
        System.out.println("Iteration: " + bp.getCurrentIteration() + " Total network error: " + bp.getTotalNetworkError());
    }
}