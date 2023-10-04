// import styling;
import './Algorithm.css';

import wheel from '../../images/wheel.png';

// import components
import Footer from '../footer.js';

// External CSS imports
import 'bootstrap/dist/css/bootstrap.css';

function Algorithm() {
    return (
        <>
        <div className = "algorithms-container">
            <div className = "piece">
                <h1 className = "header-text center-text glow-text">Algorithms</h1>
                <div className = "loop-wrapper2">
                    <div className = "wheel" id = "wheel">
                        <img src = {wheel}></img>
                    </div>
                </div>

            </div>

            <div className = "algorithms-content center">

                <div className = "divide-red"></div>

                <div className = "algorithm">
                    <h2>Neural Network:</h2>

                    <h3>We used neural networks trained on a pairwise ranking objective, which takes in each pair of racers and determines who is more likely to win. Neural networks traditionally work well for all types of machine learning tasks, and we achieve a high accuracy here.</h3>

                </div>

                <div className = "divide-red"></div>

                <div className = "algorithm">
                    <h2>XGBRanker:</h2>
                    <h3>We formulate our task as ranking - we want to find the best ranking of the racers given some prior information. We used LambdaMart, which is a classic ranking algorithm that also utilizes a pairwise ranking objective, but with gradient boosted trees, which are known to work well on tabular data tasks.</h3>

                </div>

                <div className = "divide-red"></div>

                <div className = "algorithm">
                    <h2>Random Forest:</h2>

                    <h3>We used a random forest model to classify if the driver would place top 3, and we use the probabilities of placing top 3 to rank all the drivers. Random forests, because they consist of a combination of many separate decision trees, tend to work well for tabular data tasks.</h3>

                </div>

                <div className = "divide-red"></div>

                <div className = "algorithm">
                    <h2>XGB Classifier:</h2>

                    <h3>We also used a gradient-boosted trees from the XGBoost package to predict if the driver would place top 3, using the probabilities to rank. Gradient-boosted trees, like random forests, also consist of a combination of many separate decision trees, and as such are also known to work well on tabular data tasks.</h3>

                </div>

                <div className = "divide-red space"></div>

            </div>

        <Footer/>
        </div>
        </>
    )
};

export default Algorithm;
