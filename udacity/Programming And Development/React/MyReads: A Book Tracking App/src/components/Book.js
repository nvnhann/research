import PropTypes from "prop-types";

Book.propTypes = {
    book        : PropTypes.object.isRequired,
    changeAction: PropTypes.func.isRequired,
    isSearch    : PropTypes.bool
};

const shelfOptions = [
    { id: 1, name: "Currently Reading", value: "currentlyReading"},
    { id: 2, name: "Want to Read", value: "wantToRead"},
    { id: 3, name: "Read", value: "read"},
]

export default function Book( props ){

    const { book, changeAction, isSearch }      = props;
    const { title, authors, shelf, imageLinks } = book;

    return <>
    <div className="book">
        <div className="book-top">
            <div
            className="book-cover"
            style={{
                width           : 128,
                height          : 193,
                backgroundImage : `url("${imageLinks?.smallThumbnail}")`,
            }}
            ></div>
            <div className="book-shelf-changer">
                <select
                    value={shelf}
                    onChange={(e) => changeAction(e.target.value, book)}
                >
                     {!!isSearch && (
                        <option value="none" disabled>
                            Add to...
                        </option>
                    )}

                    {!isSearch && (
                        <option value="none" disabled>
                            Move to...
                        </option>
                    )}
                    {shelfOptions.map(el => (
                            <option key={el.id} value={el.value}>
                                {el.name}
                        </option>
                    ))}
                </select>
            </div>
        </div>
        <div className="book-title">{title}</div>
        <div className="book-authors">{authors?.join(",")}</div>
    </div>
    </>
}

