import "./App.css";
import { Route, Routes } from "react-router-dom";
import HomePage from "./pages/HomePage";
import SearchPage from "./pages/SearchPage";
import { useEffect, useState } from "react";
import { getAll, update } from "./BooksAPI";

const shelves = [
  { id: 1, name: "Currently Reading", value: "currentlyReading" },
  { id: 2, name: "Want to Read", value: "wantToRead" },
  { id: 3, name: "Read", value: "read" },
];

function App() {
  const [books, setBooks] = useState([]);

  useEffect(() => {
      (async () => {
        try {
          const res = await getAll();
          setBooks(res);
        } catch (e) {
          console.warn(e);
        }
      })();
    }, []);

    const changeAction = async (shelf, book) => {
      try {
          book.shelf = shelf;
          await update(book, shelf);
          setBooks([...books.filter((b) => b.id !== book.id), book]);
      } catch (error) {
          console.warn(error)
      }
    };
  return (
    <Routes>
        <Route path="/" element={
          <HomePage 
            shelves={shelves}
            changeAction={changeAction}
            books={books}
          />
        } />
        <Route path="/search" element={
          <SearchPage
            books={books}
            changeAction={changeAction}
          />
        } />
    </Routes>
  );
}

export default App;
