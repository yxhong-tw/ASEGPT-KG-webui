/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class', // or 'media' or 'class'
  // content: {
  //   mode: 'layers',
  //   layers: ['base', 'components', 'utilities'],
  //   enabled: false,
  //   content: ['./pages/**/*.{js,ts,jsx,tsx}', './components/**/*.{js,ts,jsx,tsx}']
  // },
  content: ['./pages/**/*.{js,ts,jsx,tsx}', './components/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        oswald: ['Oswald', 'sans-serif']
      },
      inset: {
        '11/20': '55%',
        '18': '4.5rem'
      },
      margin: {
        center: '0 auto'
      },
      zIndex: {
        '60': 60,
        '100': 100
      }
    }
  },
  variants: {
    extend: {}
  },
  plugins: [
    require('@tailwindcss/forms')
  ]
}