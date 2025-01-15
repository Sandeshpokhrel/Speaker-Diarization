import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./Login.css";

function Login() {
  const [formData, setFormData] = useState({ username: "", password: "" });
  const [message, setMessage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage(null);
    setIsLoading(true);

    try {
      const response = await fetch("http://localhost:8000/auth/jwt/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Login failed");
      }

      const data = await response.json();
      localStorage.setItem("access", data.access);
      localStorage.setItem("refresh", data.refresh);

      setMessage("Login successful! Redirecting...");
      setTimeout(() => navigate("/diarization"), 1000);
    } catch (error) {
      setMessage(error.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleNavigateHome = () => {
    navigate("/");
  };

  const handleNavigateRegister = () => {
    navigate("/register");
  };

  return (
    <div className="login-page">
      <div className="home-button-container">
        <button className="home-button" onClick={handleNavigateHome}>
          Home Page
        </button>
      </div>
      <div className="login-container">
        <h2>Login</h2>
        <form onSubmit={handleSubmit}>
          <input
            name="username"
            placeholder="Username"
            onChange={handleChange}
            required
          />
          <input
            name="password"
            type="password"
            placeholder="Password"
            onChange={handleChange}
            required
          />
          <button type="submit" disabled={isLoading}>
            {isLoading ? "Logging in..." : "Login"}
          </button>
        </form>
        {message && <p className="message">{message}</p>}
        <div className="register-link">
          <span>Don't have an account? </span>
          <button className="register-button" onClick={handleNavigateRegister}>
            Register Here
          </button>
        </div>
      </div>
    </div>
  );
}

export default Login;
