import React from 'react';
import { render, screen } from '@testing-library/react';
import NotFoundPage from './404';

describe('NotFoundPage', () => {
  it('renders the correct content', () => {
    render(<NotFoundPage />);

    const headingElement = screen.getByRole('heading', { level: 2, name: /404 Page Not Found/i });
    const bodyElement = screen.getByText(/The requested page does not exist/i);

    expect(headingElement).toBeInTheDocument();
    expect(bodyElement).toBeInTheDocument();
  });
});
