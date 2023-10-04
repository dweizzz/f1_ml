import Container from 'react-bootstrap/Container';
import Nav from 'react-bootstrap/Nav';
import Navbar from 'react-bootstrap/Navbar';

import f1logo from '../../src/f1-logo.png';

function NavBar() {
  return (
    <Navbar className = "navbar" expand = "lg">
      <Container>

        <Navbar.Toggle aria-controls = "basic-navbar-nav" />
        <Navbar.Collapse id = "basic-navbar-nav">
          <Nav className = "me-auto space-between">
            <Navbar.Brand href="/" className = "no-hover"><img src = {f1logo} width="70" height="40"></img></Navbar.Brand>

            <ul className = "navbar-nav ml-auto">
              <li className = "nav-item">
                  <a className = "nav-link ml-2" href="/">
                    Home
                  </a>
              </li>
              <li className = "nav-item">
                  <a className = "nav-link ml-2" href="/predictions">
                    Predictions
                  </a>
              </li>
              <li className = "nav-item">
                  <a className = "nav-link ml-2" href="/algorithms">
                    Algorithm
                  </a>
              </li>
              <li className = "nav-item">
                  <a className = "nav-link ml-2" href="/justforfun">
                    Just for Fun!
                  </a>
              </li>
          </ul>
          </Nav>
        </Navbar.Collapse>
      </Container>
    </Navbar>
  );
}

export default NavBar;
