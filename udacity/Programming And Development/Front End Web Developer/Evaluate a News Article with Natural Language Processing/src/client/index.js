import "./styles/resets.scss";
import "./styles/base.scss";
import "./styles/header.scss";
import "./styles/footer.scss";
import "./styles/form.scss";
import { handleSubmit } from "./js/formHandler";

document.getElementById("submit").addEventListener("click", handleSubmit);

export { handleSubmit };
