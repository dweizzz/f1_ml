// import logo from './logo.svg';
import './App.css';
import f1one from './images/f1.1.jpg';

import {
  BrowserRouter as Router,
  Routes,
  Route
} from 'react-router-dom';

import Home2 from './components/pages/Home2.js';
import Predictions from './components/pages/Predictions.js';
import Algorithm from './components/pages/Algorithm.js';
import JustForFun from './components/pages/JustForFun.js';

// External CSS imports
import 'bootstrap/dist/css/bootstrap.css';


// Internal component imports
import NavBar from './components/NavBar.js';
import Footer from './components/footer.js';

function App() {
  return (
      <Router>
        <head>
          <link rel="preconnect" href="https://fonts.googleapis.com" />
          <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
          <link href="https://fonts.googleapis.com/css2?family=Oswald:wght@700&display=swap" rel="stylesheet" />
        </head>
        <NavBar />
        <Routes>
          <Route exact path='/' element={<Home2 />} />
          <Route exact path='/predictions' element={<Predictions />}/>
          <Route exact path='/algorithms' element={<Algorithm />}/>
          <Route exact path='/justforfun' element={<JustForFun />}/>

        </Routes>
      </Router>

  );
}

export default App;
