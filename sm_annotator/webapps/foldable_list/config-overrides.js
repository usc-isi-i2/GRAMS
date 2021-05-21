module.exports = {
  webpack: function (config, env) {
    // update the configuration to change the way .css file is process
    // instead of using style-loader to load to style element, we need to load it to string
    // the current code ['to-string-loader', 'css-loader'] only works for css-loader 3.4.2 which is used by
    // create-react-app. It doesn't work for css-loader 4.2.2.
    // if you need to update this in the future, you can eject the project to see which configuration you need
    // to modify!

    config.module.rules[2].oneOf = config.module.rules[2].oneOf.map(rule => {
      if (!rule.hasOwnProperty('use')) return rule;

      return Object.assign({}, rule, {
        use: rule.use.map(options => /style-loader/.test(typeof options === 'string' ? options : options.loader)
          ? { loader: require.resolve('to-string-loader'), options: {} }
          : options)
      });
    });

    // let rules = config.module.rules[2].oneOf;
    // for (let i = 0; i < rules.length; i++) {
    //   if (!(rules[i].test instanceof RegExp)) {
    //     continue;
    //   }

    //   if (rules[i].test.test("styles.css") && (rules[i].exclude !== undefined && !rules[i].exclude.test("styles.css"))) {
    //     rules[i].use = ['to-string-loader', 'css-loader'];
    //   }
    // }
    return config;
  }
}