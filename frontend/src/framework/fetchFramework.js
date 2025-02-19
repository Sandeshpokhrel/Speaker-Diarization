import {backendUrl } from "../appConstants.js";

export default async function fetchFramework({ endpoint, body, form }) {
  endpoint = endpoint[0] === "/" ? endpoint.slice(1) : endpoint;
  let target = backendUrl.endsWith("/")
    ? `${backendUrl}${endpoint}`
    : `${backendUrl}/${endpoint}`;
  try {
    if (body) {
      const response = await fetch(target, {
        method: "POST",
        credentials: "include",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
      });
      const res = await response.json();
      return res;
    } else if (form) {
      const response = await fetch(target, {
        body: form,
        credentials: "include",
        method: "POST",
      });
      const res = await response.json();
      return res;
    } else {
      const response = await fetch(target, {
        method: "GET",
        credentials: "include",
      });
      const res = await response.json();
      return res;
    }
  } catch (err) {
    console.error(err);
    return {
      error: true,
      message: `Error fetching data: ${err.message}`,
    };
  }
}
