/* Create an array named products which you will use to add all of your product object literals that you create in the next step. */
let products = [
  {
      productId: 1,
      image: "images/cherry.jpg",
      name: "Cherry",
      price: 12,
      quantity: 0
  },
  {
      productId: 2,
      image: "images/orange.jpg",
      name: "Orange",
      price: 13,
      quantity: 0
  },
  {
      productId: 3,
      image: "images/strawberry.jpg",
      name: "Strawberry",
      price: 14,
      quantity: 0
  },
];
/* Create 3 or more product objects using object literal notation 
   Each product should include five properties
   - name: name of product (string)
   - price: price of product (number)
   - quantity: quantity in cart should start at zero (number)
   - productId: unique id for the product (number)
   - image: picture of product (url string)
*/

/* Images provided in /images folder. All images from Unsplash.com
   - cherry.jpg by Mae Mu
   - orange.jpg by Mae Mu
   - strawberry.jpg by Allec Gomes
*/

/* Declare an empty array named cart to hold the items in the cart */
let cart = [];

let totalPaid = 0;
/* Create a function named addProductToCart that takes in the product productId as an argument
  - addProductToCart should get the correct product based on the productId
  - addProductToCart should tlet totalPaid = 0;
hen increase the product's quantity
  - if the product is not already in the cart, add it to the cart
*/
const addProductToCart = (productId) => {
  // Find the index of the product in the cart array, based on its ID
  const idx = cart.findIndex((cartItem) => cartItem.productId === productId);
  // Retrieve the product details from wherever they are stored (e.g., products array)
  const product_idx = products.findIndex((p) => p.productId === productId);
  const product = products.find((p) => p.productId === productId);
  // If the product is not found, return or handle the error accordingly
  if (product_idx === -1) {
    return;
  }

  if (idx >= 0) {
    // If the product is already in the cart, check if quantity is less than available stock
      cart[idx].quantity += 1;
  } else {
    cart.push({...product, quantity: 1});
  }
  products[product_idx].quantity += 1;

};


/* Create a function named increaseQuantity that takes in the productId as an argument
  - increaseQuantity should get the correct product based on the productId
  - increaseQuantity should then increase the product's quantity
*/

const increaseQuantity = productId => {
    // Find the index of the product in the cart array, based on its ID
    const idx = cart.findIndex((cartItem) => cartItem.productId === productId);
  
    // Retrieve the product details from wherever they are stored (e.g., products array)
    const product_idx = products.findIndex((p) => p.productId === productId);
      // If the product is not found, return or handle the error accordingly
  if (product_idx == -1) {
    return;
  }
    if( idx >=0 ){
      cart[idx].quantity += 1;
      products[product_idx].quantity += 1;
    }
}

/* Create a function named decreaseQuantity that takes in the productId as an argument
  - decreaseQuantity should get the correct product based on the productId
  - decreaseQuantity should decrease the quantity of the product
  - if the function decreases the quantity to 0, the product is removed from the cart
*/

const decreaseQuantity = productId => {
    // Find the index of the product in the cart array, based on its ID
    const idx = cart.findIndex((cartItem) => cartItem.productId === productId);
  
    // Retrieve the product details from wherever they are stored (e.g., products array)
    const product_idx = products.findIndex((p) => p.productId === productId);
      // If the product is not found, return or handle the error accordingly
  if (product_idx == -1) {
    return;
  }
    if( idx >=0 ){
      if( Number(cart[idx].quantity) === 1) return removeProductFromCart(productId)
      cart[idx].quantity -= 1;
      products[product_idx].quantity -= 1;
    }
}

/* Create a function named removeProductFromCart that takes in the productId as an argument
  - removeProductFromCart should get the correct product based on the productId
  - removeProductFromCart should update the product quantity to 0
  - removeProductFromCart should remove the product from the cart
*/

const removeProductFromCart = productId =>{
    const idx = cart.findIndex((cartItem) => cartItem.productId === productId);
    const product_idx = products.findIndex((p) => p.productId === productId);

    if (idx !== -1) {
      cart_new = cart.filter(c => c.productId !== productId);
      cart = cart_new
      products[product_idx].quantity = 0;
    }
    else{
      // You may add additional logic here, like showing an error message to the user.
    }
}


/* Create a function named cartTotal that has no parameters
  - cartTotal should iterate through the cart to get the total of all products
  - cartTotal should return the sum of the products in the cart
*/
const cartTotal = () => {
  const total = cart.reduce((accumulator, cartItem) => {
    return accumulator + cartItem.price * cartItem.quantity;
  }, 0);
  return total;
}

/* Create a function called emptyCart that empties the products from the cart */
const emptyCart = () => {
  cart = [];
}
/* Create a function named pay that takes in an amount as an argument
  - pay will return a negative number if there is a remaining balance
  - pay will return a positive number if money should be returned to customer
*/

const pay = (amount) => {
  // Assuming the amount passed to the function represents the total cost of products purchased
  // You can adjust the logic based on your specific use case
  totalPaid += amount;

  const totalCartValue = cartTotal(); // You can use the previously defined cartTotal function or implement it here

  // Calculate the difference between the total cart value and the amount paid by the customer
  const remainingBalance = totalPaid - totalCartValue;

  return remainingBalance;
}

/* Place stand out suggestions here (stand out suggestions can be found at the bottom of the project rubric.)*/


/* The following is for running unit tests. 
   To fully complete this project, it is expected that all tests pass.
   Run the following command in terminal to run tests
   npm run test
*/

module.exports = {
  products,
  cart,
  addProductToCart,
  increaseQuantity,
  decreaseQuantity,
  removeProductFromCart,
  cartTotal,
  pay, 
  emptyCart,
  /* Uncomment the following line if completing the currency converter bonus */
  // currency
}