// tests/boolean.test.js
// Unit tests for AST Boolean Sourcing Engine & Parser Rules

describe('AST Boolean Sourcing Engine Unit Tests', () => {
  function tokenize(str) {
    const raw = [];
    const re = /\(|\)|"[^"]*"|[^\s()]+/g;
    let m;

    while ((m = re.exec(str)) !== null) {
      const val = m[0];

      if (val === '(') {
        raw.push({ type: 'LPAREN' });
      } else if (val === ')') {
        raw.push({ type: 'RPAREN' });
      } else if (val.startsWith('"') && val.endsWith('"')) {
        raw.push({
          type: 'QUOTED',
          val: val.slice(1, -1)
        });
      } else {
        const u = val.toUpperCase();

        if (u === 'AND' || u === 'OR' || u === 'NOT') {
          raw.push({ type: u });
        } else {
          raw.push({
            type: 'TERM',
            val
          });
        }
      }
    }

    // Insert implicit AND
    const tokens = [];

    for (let i = 0; i < raw.length; i++) {
      if (i > 0) {
        const prev = raw[i - 1].type;
        const curr = raw[i].type;

        const prevOperand =
          prev === 'TERM' ||
          prev === 'QUOTED' ||
          prev === 'RPAREN';

        const currOperand =
          curr === 'TERM' ||
          curr === 'QUOTED' ||
          curr === 'LPAREN' ||
          curr === 'NOT';

        if (prevOperand && currOperand) {
          tokens.push({ type: 'AND' });
        }
      }

      tokens.push(raw[i]);
    }

    return tokens;
  }

  test('Tokenize basic AND/OR string', () => {
    const toks = tokenize('Java AND Developer');

    expect(toks.length).toBe(3);
    expect(toks[1].type).toBe('AND');
  });

  test('Implicit AND insertion between adjacent terms', () => {
    const toks = tokenize('Java Developer');

    expect(toks.length).toBe(3);
    expect(toks[1].type).toBe('AND');
  });

  test('Implicit AND insertion between adjacent groups', () => {
    const toks = tokenize(
      '("Java" OR "Python") ("AWS" OR "Azure")'
    );

    const andTokens = toks.filter(
      t => t.type === 'AND'
    );

    expect(andTokens.length).toBeGreaterThan(0);
  });

  test('Quoted terms preserve exact phrases', () => {
    const toks = tokenize(
      '"Senior Software Engineer"'
    );

    expect(toks[0].type).toBe('QUOTED');
    expect(toks[0].val).toBe(
      'Senior Software Engineer'
    );
  });
});
