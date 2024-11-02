import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { search } from "../BooksAPI";
import Book from "../components/Book";
import PropTypes from "prop-types";

SearchPage.propTypes = {
    books       : PropTypes.array.isRequired,
    changeAction: PropTypes.func.isRequired
} 

export default function SearchPage( props ){
    
    const { books, changeAction } = props;
    const [query, setQuery] = useState("");
    const [bookList, setBookList] = useState([]);


    const shelves = (BookSeach, BookList) => {
        return BookSeach.map((book) => {
            const matchingBook = BookList.find((b) => b.id === book.id);
            const shelf = matchingBook ? matchingBook.shelf : "none";
            return { ...book, shelf };
        });
    };

    useEffect(()=>{
        ( async () => {
            if(query.length > 0){
                try {
                    const res = await search(query)
                    if (res.error) return setBookList([]);
                    setBookList(shelves(res, books));
                } catch (error) {
                    console.warn(error);
                }
            } else {
                setBookList([]);
            }
        })()
    }, [query, books])

    return (
        <div className="search-books">
        <div className="search-books-bar">
            <Link className="close-search" to="/">Close</Link>
            <div className="search-books-input-wrapper">
            <input
                type="text"
                placeholder="Search by title, author, or ISBN"
                value={query}
                onChange={e => setQuery(e.target.value)}
            />
            </div>
        </div>

        {!!query && (
            <div className="search-books-results">
                <ol className="books-grid">
                    {bookList?.map( book => (
                        <li key={book?.id}>
                            <Book
                                book={book}
                                changeAction={changeAction}
                                isSearch={true}
                            />
                        </li>
                    ))}
                </ol>
            </div>
        )}
        </div>
    )
}