import SearchButton from "../components/SearchButton";
import BookList from "../components/BookList";
import PropTypes from "prop-types";

HomePage.propTypes = {
    shelves     : PropTypes.array.isRequired,
    changeAction: PropTypes.func.isRequired,
    books       : PropTypes.array.isRequired
}
export default function HomePage( props ){
    const { shelves, changeAction, books } = props;
    return (
        <div className="list-books">
            <div className="book-shelf">
                <div className="list-books-title">
                <h1>MyReads</h1>
                </div>
                <div className="list-books-content">
                <div>
                    {shelves.map((shelf, key) => (
                        <BookList
                            key={key}
                            title={shelf.name}
                            books={
                                books &&
                                books.filter((book) => book && book.shelf === shelf.value)
                            }
                            changeAction={changeAction}
                        />
                    ))}
                </div>
                </div>
            </div>
            <SearchButton />
        </div>
    )
}