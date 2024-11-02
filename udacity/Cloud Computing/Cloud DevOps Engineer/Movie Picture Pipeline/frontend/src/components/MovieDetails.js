import React, { useState, useEffect } from 'react';
import axios from 'axios';

function MovieDetail({ movie }) {
  const [details, setDetails] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchMovieDetails = async () => {
      setIsLoading(true);
      try {
        const response = await axios.get(`${process.env.REACT_APP_MOVIE_API_URL}/movies/${movie.id}`);
        setDetails(response.data);
      } catch (err) {
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    };

    if (movie.id) {
      fetchMovieDetails();
    }
  }, [movie.id]);

  if (isLoading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div>
      {details && (
        <>
          <h2>{details.movie.title}</h2>
          <p>{details.movie.description}</p>
        </>
      )}
    </div>
  );
}

export default MovieDetail;
