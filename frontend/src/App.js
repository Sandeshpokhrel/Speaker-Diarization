import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import HomePage from "./components/HomePage";
import RegisterPage from "./components/Register";
import LoginPage from "./components/Login";
import UserDetailPage from "./components/UserDetails";
import MainPage from "./components/MainPage";
import AboutPage from "./components/AboutPage";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/userdetails" element={<UserDetailPage />} />
        <Route path="/diarization" element={<MainPage />} />
        <Route path="/about" element={<AboutPage />} />
      </Routes>
    </Router>
  );
}

export default App;
