// import styling;
import './Home2.css';

// import images
import f1one from '../../images/f1NOTROYALTYFREE.png';
import selfie from '../../images/selfie.jpeg';

// import components
import Footer from '../footer.js';

// External CSS imports
import 'bootstrap/dist/css/bootstrap.css';

function Home2() {
    return (
        <>
        <div className = "home-container">
            <div className = "image-container center-text">
                <img src = {f1one} alt = "F1 Car" />
                <h1 className = "image-title reset center no-scroll">F1 ML</h1>
            </div>

            <div className = "about-us piece center rows">
                <div className = "left-img">
                    <img src = {selfie} />
                </div>
                <div className = "right-text">
                    <h3 className = "text-underline">Welcome to our website!</h3>

                    <h4>We are a group of Cornell students who love to work with data, and we created this project as members of CDS—Cornell Data Science!</h4>

                    <h4>As F1 fans, we've decided to look into predicting the next F1 race winner. This website documents our journey: the models we decided to use, our betting logs, and how our predictions have improved.</h4>

                </div>
            </div>

            <div className = "our-work center">
                <h1 className = "our-work-title center-text">Our Work</h1>

                <div className = "blocks">
                    <div className = "prediction block">
                        <h3>Predictions</h3>
                        <h4>Explore our predictions page to see how we've been progressing as the races go by!
                        We've created a betting log chart, a table of our predictions, and we are working
                        on displaying the error of our predictions.</h4>
                        <a href = "/predictions">
                            <button type = "button" className = "new-button">Learn more! &rarr;</button>
                        </a>
                    </div>

                    <div className = "divide-red center"></div>

                    <div className = "algorithm block">
                        <h3>Algorithm</h3>
                        <h4>On this page, we provide insight on the algorithms we've used to create our predictions.
                        We detail why we used each model, along with how these models compare depending
                        on the nature of the model.</h4>
                        <a href = "/algorithm" className = "new-button">
                            <button type = "button" className = "new-button">Learn more! &rarr;</button>
                        </a>
                    </div>

                    <div className = "divide-red center"></div>

                    <div className = "just-for-fun block">
                        <h3>Just for fun</h3>
                        <h4>This page is just for fun! Our web developers have created a countdown timer to the next race,
                        so feel free to visit this one to check it out.
                        More ideas are coming—be sure to come back before the next race!</h4>
                        <a href = "/justforfun" className = "new-button">
                            <button type = "button" className = "new-button">Learn more! &rarr;</button>
                        </a>
                    </div>

                    <div className = "divide-red center"></div>

                </div>
            </div>

            <div className = "subteams center piece">
                <h1>Our Subteams</h1>
                <div className = "rows">

                    <div className = "cards equal-boxes">
                        <h2>Front-end</h2>
                        <div className = "divide-red center"></div>
                        <p>Our front-end team is charge of the website! Working hard on design,
                        they've built this site using React, Bootstrap, and various
                        other packages. Apart from this, they've also created
                        visualizations: betting logs, racer comparisons, and confidence intervals
                        of our predictions.
                        </p>
                    </div>

                    <div className = "cards equal-boxes">
                        <h2>Data engineering</h2>
                        <div className = "divide-red center"></div>
                        <p>The data engineering
                        team's role is gathering the data we need for the MLE team to work with. This relates to
                        webscraping, where we gather data from this season's races to add to our
                        full database. As for the prevoius rce data, the DE team utilized
                        an API.
                        </p>
                    </div>

                    <div className = "cards equal-boxes">
                        <h2>Machine learning</h2>
                        <div className = "divide-red center"></div>
                        <p>The MLE
                        team is charge of deciding which models we want to train, along with actually training
                        the models. Currently, we have three models: lasso, decision tree, and XGBoost. We want
                        multiple models to compare each output and determine the racer most likely to win the
                        next race!
                        </p>
                    </div>

                </div>
            </div>

            <Footer/>

        </div>
        </>
    )
};

export default Home2;
