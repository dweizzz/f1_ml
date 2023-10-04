import './JustForFun.css';
import 'bootstrap/dist/css/bootstrap.css';

// import components
import Footer from '../footer.js';

// Internal component imports
import React, { createElement } from 'react';
// import ReactDOM from 'react-dom';
import Countdown, { zeroPad} from 'react-countdown';

function Counts() {
    // Find next race from array of race dates
    const Completionist = () => <span>The race has begun!</span>;
    const raceDates = [new Date('2023-03-19T13:00:00'), new Date('2023-04-02T01:00:00'),
                    new Date('2023-04-30T07:00:00'), new Date('2023-05-07T15:30:30'),
                    new Date('2023-05-21T09:00:00'), new Date('2023-05-28T09:00:00'),
                    new Date('2023-06-04T09:00:00'), new Date('2023-06-18T14:00:00'),
                    new Date('2023-07-02T09:00:00'), new Date('2023-07-09T10:00:00'),
                    new Date('2023-07-23T09:00:00'), new Date('2023-07-30T09:00:00'),
                    new Date('2023-08-27T09:00:00'), new Date('2023-09-03T09:00:00'),
                    new Date('2023-09-17T08:00:00'), new Date('2023-09-24T01:00:00'),
                    new Date('2023-10-08T13:00:00'), new Date('2023-10-22T15:00:00'),
                    new Date('2023-10-29T16:00:00'), new Date('2023-11-05T12:00:00'),
                    new Date('2023-11-19T01:00:00'), new Date('2023-11-26T08:00:00')
                    ]
    var smallestDiff = new Date('2023-11-26T08:00:00') - Date.now();
    var currentDiff;
      for (let i = 0; i < raceDates.length; i++) {
          if (Date.now() < raceDates[i]) {
              currentDiff = raceDates[i] - Date.now();
              if (currentDiff <= smallestDiff) {
                  smallestDiff = currentDiff;
                  var upcomingData = raceDates[i];
              }
          }
      }

    const renderer = ({ days, hours, minutes, seconds, completed }) => {
    if (completed) {
      // Render a completed state
      return <Completionist />;
    } else {
      // Render a countdown
      return <span className = "times">
        <div className = "time-colon">
          <div className = "time-pair">
            <div className = "time-num">{zeroPad(days)}</div>
            <div className = "time-val">days</div>
          </div>
          <div className = "colon">:</div>
        </div>
        <div className = "time-colon">
          <div className = "time-pair">
            <div className = "time-num">{zeroPad(hours)}</div>
            <div className = "time-val">hours</div>
          </div>
          <div className = "colon">:</div>
        </div>
        <div className = "time-colon">
          <div className = "time-pair">
            <div className = "time-num">{zeroPad(minutes)}</div>
            <div className = "time-val">minutes</div>
          </div>
          <div className = "colon">:</div>
        </div>
        <div className = "time-colon">
          <div className = "time-pair">
            <div className = "time-num">{zeroPad(seconds)}</div>
            <div className = "time-val">seconds</div>
          </div>
        </div>
      </span>;
    }};

    return(
        createElement(
          'div',
          {},
          <Countdown date = {upcomingData} renderer = {renderer} />
        )
      )
    }

function JustForFun() {
    return (
        <>
        <div className></div>
        <div className = "countdown piece">
        <h2>Countdown to the Next Race</h2>
            <div id = "pleases">
              <Counts />
            </div>
        </div>
        <Footer/>
        </>
    )
}

export default JustForFun;
