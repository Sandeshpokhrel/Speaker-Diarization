import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./Register.css";

function Register() {
  const [formData, setFormData] = useState({
    username: "",
    password: "",
    email: "",
    first_name: "",
    last_name: "",
    phone_number: "",
  });
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
      const response = await fetch("http://localhost:8000/auth/users/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Registration failed");
      }

      setMessage("Registration successful! Please log in.");
      setTimeout(() => navigate("/login"), 1500);
    } catch (error) {
      setMessage(error.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleNavigateHome = () => {
    navigate("/");
  };

  const handleNavigateLogin = () => {
    navigate("/login");
  };

  return (
    <div className="register-page">
      <div className="home-button-container">
        <button className="home-button" onClick={handleNavigateHome}>
          Home Page
        </button>
      </div>
      <div className="register-container">
        <h2>Register</h2>
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
          <input
            name="email"
            type="email"
            placeholder="Email"
            onChange={handleChange}
            required
          />
          <input
            name="first_name"
            placeholder="First Name"
            onChange={handleChange}
          />
          <input
            name="last_name"
            placeholder="Last Name"
            onChange={handleChange}
          />
          <input
            name="phone_number"
            placeholder="Phone Number"
            onChange={handleChange}
          />
          <button type="submit" disabled={isLoading}>
            {isLoading ? "Registering..." : "Register"}
          </button>
        </form>
        {message && <p className="message">{message}</p>}
        <div className="login-link">
          <span>Already have an account? </span>
          <button className="login-button" onClick={handleNavigateLogin}>
            Login
          </button>
        </div>
      </div>
    </div>
  );
}

export default Register;
