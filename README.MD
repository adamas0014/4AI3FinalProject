<h1>4AI3 - Induction Motor Fault</h1>
<h3><i>Myko Smid 400192728, Adam Sokacz 400177446</i></h3>
<br />
<h2> Aggregating Data </h2>
The following preprocessing code aggregates data from an induction fault dataset prior to preprocessing it. It does this using the glob python library, that parses a folder and returns a list of every path that matches a specified filter.
<br />
<h2> Downsampling Data </h2>
THe downsampling process iterates through the master dataframe and averages a specified number of samples. This allows a single data point to represent 200 datapoints. The benefits of downsampling is it smooths the data and also makes it easier to work with within Python.
<br />
<h2>Neural Net Hyperparameter Tuning</h2>
Using a predefined dictionary of hyperparameter profiles, a function was written to iterate through and build a neural network model using each profile, and compare them in terms of accuracy.




