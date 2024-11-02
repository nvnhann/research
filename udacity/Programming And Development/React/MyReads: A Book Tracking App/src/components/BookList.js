import PropTypes from "prop-types";
import Book from "./Book";

BookList.propTypes = {
    books       : PropTypes.array.isRequired,
    title       : PropTypes.string.isRequired,
    changeAction: PropTypes.func.isRequired
}

export default function BookList( props ){

    const { books, title, changeAction } = props;

    return <>
        <div className="bookshelf">
            <h2 className="bookshelf-title">{title}</h2>
            <div className="bookshelf-books">
                <ol className="books-grid">
                    {books?.map( book => (
                    <li key={book.id}>
                        <Book 
                            book={book}
                            changeAction={changeAction}
                        />
                    </li>
                    ))}
                </ol>
            </div>
        </div>
    </>
}