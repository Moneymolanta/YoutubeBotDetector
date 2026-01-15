/**
 * Utility functions for handling HTML content
 */

import parse from 'html-react-parser';

/**
 * Decodes HTML entities in a string and returns a React element or string
 * @param htmlString - String containing HTML entities to decode
 * @returns React element or string with decoded HTML entities
 */
export function decodeHtml(htmlString: string | null | undefined): any {
  if (!htmlString) return '';
  return parse(htmlString);
}

/**
 * Decodes HTML entities in a string and returns only a string (not React elements)
 * Useful for cases where React elements can't be used (like title attributes)
 * @param htmlString - String containing HTML entities to decode
 * @returns String with decoded HTML entities
 */
export function decodeHtmlToString(htmlString: string | null | undefined): string {
  if (!htmlString) return '';
  
  // Create a temporary DOM element
  const txt = document.createElement('textarea');
  txt.innerHTML = htmlString;
  
  // Use the browser's built-in HTML entity decoding
  return txt.value;
}
