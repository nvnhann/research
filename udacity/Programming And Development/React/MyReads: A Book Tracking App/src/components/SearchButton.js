import { Link } from "react-router-dom";

export default function SearchButton () {
    return (
        <div className="open-search">
            <Link to="/search"></Link>
        </div>
    )
}