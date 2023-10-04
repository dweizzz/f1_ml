// import chart from google charts
import { Chart } from "react-google-charts";
import React, { createElement } from 'react';
import { render } from "react-dom";

function BetsMoney() {
    const data = [
        ["race", "Kelly Criterion", "Naive", "Travis"],
        ["start", 0, 0, 0],
        ["1", -0.0929259, 0.0191352, 0.0291683],
        ["2", 0.130392,  0.0464564, -0.000975991],
        ["3", 0, 0, 0],
        ["4", 0, 0, 0]
    ];

    const options = {
        title: "betting log",
        legend: { position: "bottom" },
        curveType: 'function',
        vAxis: {format: '#%'},
        backgroundColor: 'white',
      };

    return(
        createElement(
            'div',
            {},
            <Chart chartType = "LineChart"
                    width = "700px"
                    height ="400px"
                    data = {data}
                    options = {options}
                    renderer = {render}
                    className = "linegraph"
                  />
        )
      )
}

export default BetsMoney;
