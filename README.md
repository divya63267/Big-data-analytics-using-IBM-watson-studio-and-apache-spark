# Big-Data-Analytics-with-Apache-Spark-and-IBM-Watson-Studio
 IBM Watson Studio with Apache Spark for big data analytics. Big-Data-Analytics-with-Apache-Spark-and-IBM-Watson-Studio
First run the exploratory nodebook first. Once Complete, run the data processing notebook.

> Note: Running the exploratory notebook first is a requirement. It loads libraries and packages that are required in the data processing notebook.

When a notebook is executed, what is actually happening is that each code cell in
the notebook is executed, in order, from top to bottom.

Each code cell is selectable and is preceded by a tag in the left margin. The tag
format is `In [x]:`. Depending on the state of the notebook, the `x` can be:

* A blank, this indicates that the cell has never been executed.
* A number, this number represents the relative order this code step was executed.
* A `*`, this indicates that the cell is currently executing.

There are several ways to execute the code cells in your notebook:

* One cell at a time.
  * Select the cell, and then press the `Play` button in the toolbar.
* Batch mode, in sequential order.
  * From the `Cell` menu bar, there are several options available. For example, you
    can `Run All` cells in your notebook, or you can `Run All Below`, that will
    start executing from the first cell under the currently selected cell, and then
    continue executing all cells that follow.
* At a scheduled time.
  * Press the `Schedule` button located in the top right section of your notebook
    panel. Here you can schedule your notebook to be executed once at some future
    time, or repeatedly at your specified interval.

### 4. Save and Share

#### How to save your work:

Under the `File` menu, there are several ways to save your notebook:

* `Save` will simply save the current state of your notebook, without any version
  information.
* `Save Version` will save your current state of your notebook with a version tag
  that contains a date and time stamp. Up to 10 versions of your notebook can be
  saved, each one retrievable by selecting the `Revert To Version` menu item.

#### How to share your work:

You can share your notebook by selecting the `Share` button located in the top
right section of your notebook panel. The end result of this action will be a URL
link that will display a “read-only” version of your notebook. You have several
options to specify exactly what you want shared from your notebook:

* `Only text and output`: will remove all code cells from the notebook view.
* `All content excluding sensitive code cells`:  will remove any code cells
  that contain a *sensitive* tag. For example, `# @hidden_cell` is used to protect
  your credentials from being shared.
* `All content, including code`: displays the notebook as is.
* A variety of `download as` options are also available in the menu.

### 5. Explore and Analyze the Data

Both notebooks are well documented and will guide you through the exercise. Some of the main tasks that will be covered include:

* Load packages and data and do the initial transformation and various feature engineering.

* Sample the dataset and use the powerful ggplot2 library from R to do various exploratory analysis.

* Run PCA (Principal Component Analysis) to reduce the dimensions of the dataset and select the k components to cover 90% of variance.

You will also see the advantages of using R4ML, which is a git-downloadable open-source R packaged from IBM. Some of these include:

* Created on top of SparkR and Apache SystemML, so it supports features from both.

* Acts as an R bridge between SparkR and Apache SystemML.

* Provides a collection of canned algorithms.

* Provides the ability to create custom ML algorithms.

* Provides both SparkR and Apache SystemML functionality.

* APIs that should be familiar to R users.

## Sample output

The following screen-shots shows the histogram of the exploratory analysis.

![Exploratory Analysis Histogram](doc/source/images/r4ml-hist.png)

The following screen-shots shows the correlation between various features of the exploratory analysis.

![Exploratory Analysis Correlation between various features](doc/source/images/r4ml-corr.png)

The following screen-shots shows the output of the dimensionality reduction using PCA and how only 6 components of PCA carries 90% of information.

![Dimension Reduction using PCA](doc/source/images/r4ml-pca-dimred.png)

Awesome job following along! Now go try and take this further or apply it to a different use case!

## Links

* [Data Set](http://stat-computing.org/dataexpo/2009/the-data.html)

## Learn more

* **Data Analytics Code Patterns**: Enjoyed this Code Pattern? Check out our other [Data Analytics Code Patterns](https://developer.ibm.com/technologies/data-science/)
* **AI and Data Code Pattern Playlist**: Bookmark our [playlist](https://www.youtube.com/playlist?list=PLzUbsvIyrNfknNewObx5N7uGZ5FKH0Fde) with all of our Code Pattern videos
* **Watson Studio**: Master the art of data science with IBM's [Watson Studio](https://dataplatform.cloud.ibm.com/)

## License

This code pattern is licensed under the Apache License, Version 2. Separate third-party code objects invoked within this code pattern are licensed by their respective providers pursuant to their own separate licenses. Contributions are subject to the [Developer Certificate of Origin, Version 1.1](https://developercertificate.org/) and the [Apache License, Version 2](https://www.apache.org/licenses/LICENSE-2.0.txt).

[Apache License FAQ](https://www.apache.org/foundation/license-faq.html#WhatDoesItMEAN)
