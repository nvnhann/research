import { render, fireEvent } from '@testing-library/react';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import React from 'react';
import NewPollPage from './NewPollPage';
import { configureStore } from '@reduxjs/toolkit';
import userReducer from "../store/userSlice";

describe('NewPollPage', () => {

    const rootReducer = {
        user: userReducer,
    };
    const store = configureStore({
        reducer: rootReducer,
    });

  it('should render the component', () => {
    const component = render(
      <Provider store={store}>
        <BrowserRouter>
          <NewPollPage />
        </BrowserRouter>
      </Provider>
    );
    expect(component).toBeDefined();
    expect(component).toMatchSnapshot();
  });

  it('should submit a new poll', () => {
    const component = render(
      <Provider store={store}>
        <BrowserRouter>
          <NewPollPage />
        </BrowserRouter>
      </Provider>
    );

    const firstOptionInput = component.getByLabelText('First Option');
    const secondOptionInput = component.getByLabelText('Second Option');
    const submitButton = component.getByText('Submit');

    fireEvent.change(firstOptionInput, { target: { value: 'Option A' } });
    fireEvent.change(secondOptionInput, { target: { value: 'Option B' } });
    fireEvent.click(submitButton);
    expect(component).toMatchSnapshot();
  });
});
