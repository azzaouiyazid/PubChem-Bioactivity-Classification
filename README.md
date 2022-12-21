# PubChem-Bioactivity-Classification
<h1>README</h1>
<p>Welcome to the PubChem Bioactivity Classification project! This project aims to classify molecular bioactivity as active or inactive using machine learning and deep learning techniques.</p>
<h2>Project Objectives</h2>
<p>The main objective of this project is to build and evaluate ML and DL models that can predict the bioactivity of molecules based on their chemical structure. To achieve this, we will follow the following steps:</p>
<ol>
  <li>Read and clean the data: The raw PubChem bioactivity data will be read into a data frame and cleaned by skipping rows that do not contain data and deleting NaN and duplicate values in the 'PUBCHEM_EXT_DATASOURCE_SMILES' column.</li>
  <li>Describe the molecular structure and the binary classification: The molecular structure will be described using the MHFP (Molecular Hash Fingerprint) method, which calculates a fingerprint for each molecule. The 'PUBCHEM_ACTIVITY_SCORE' column will be converted to a binary activity and set as the target variable (y).</li>
  <li>Prepare the data for machine learning and deep learning: The data will be split into training and test sets and scaled using appropriate techniques.</li>
  <li>Machine learning: At least three scikit-learn ML models will be tried and the best model will be selected using cross-validation. The parameters of the best model will be optimized using grid search.</li>
  <li>Deep learning: A deep neural network (DNN) will be built, compiled, and fit to the data. The learning curve (loss vs. epoch) will be plotted and the performance of the model will be optimized by varying the number of layers, the number of neurons per layer, the dimensionality of the MHFP, and/or the number of epochs.</li>
  <li>Evaluate the models: The performance of the ML and DL models will be evaluated using appropriate metrics, such as accuracy, precision, and recall.</li>
</ol>
<h2>Data Preprocessing</h2>
<p>Before we can build and evaluate the ML and DL models, we need to prepare the data by performing the following preprocessing steps:</p>
<ol>
  <li>Read the data into a data frame.</li>
  <li>Skip rows that do not contain data.</li>
  <li>Use only the columns 'PUBCHEM_EXT_DATASOURCE_SMILES' and 'PUBCHEM_ACTIVITY_SCORE'.</li>
  <li>Delete NaN and remove duplicate data in 'PUBCHEM_EXT_DATASOURCE_SMILES'.</li>

  <li>Use MHFP to calculate the fingprint and reformat it into a data frame that can be used as an X.</li>
  <li>Convert "PUBCHEM_ACTIVITY_SCORE" to a binary activity and set it to y.</li>
  <li>Split the data into training and test sets.</li>
  <li>Scale the features using appropriate techniques.</li>
</ol>
<h2>Machine Learning</h2>
<p>Once the data is prepared, we can start building and evaluating ML models. We will use scikit-learn to try at least three different ML models and use cross-validation to select the best model. Then, we will use grid search to optimize the parameters of the best model.</p>
<h2>Deep Learning</h2>
<p>After selecting the best ML model, we will build, compile, and fit a DNN to the data. We will plot the learning curve (loss vs. epoch) to visualize the training process. We will also vary the number of layers, the number of neurons per layer, the dimensionality of the MHFP, and/or the number of epochs to optimize the performance of the model.</p>
<p>To run the code, you will need to have Python 3 and the following libraries installed:</p>

<ul>
  <li>pandas</li>
  <li>numpy</li>
  <li>scikit-learn</li>
  <li>tensorflow (for deep learning)</li>
</ul>
<p>You can install these libraries using <code>pip install</code>.</p>
<p>To run the code, clone the repository and navigate to the project directory. Then, run the <code>main.py</code> script:</p>
<pre>
git clone https://github.com/azzaouiyazid/PubChem-Bioactivity-Classification.git
cd PubChem-Bioactivity-Classification
python main.py
</pre>
<p>The code will execute the project milestones in order and output the results.</p>
<h2>Additional Notes</h2>
<ul>
  <li>The raw data can be downloaded from Data folder.</li>
  </li>
  <li>The scikit-learn and tensorflow libraries provide a wide range of machine learning and deep learning models that you can experiment with. Try out different models and parameters to see which ones work best for your data.</li>
  <li>Make sure to tune the hyperparameters of the models to get the best performance. You can use techniques such as grid search or random search to optimize the hyperparameters.</li>
  <li>Don't forget to evaluate the models using appropriate metrics and compare the results to choose the best model for your data.</li>
</ul>
