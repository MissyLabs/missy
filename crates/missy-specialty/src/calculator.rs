use anyhow::{Result, bail, ensure};
struct Parser<'a> {
    s: &'a [u8],
    pos: usize,
    nodes: usize,
}
pub fn calculate(expression: &str) -> Result<f64> {
    ensure!(
        !expression.is_empty() && expression.len() <= 4096 && expression.is_ascii(),
        "arithmetic expression size/encoding denied"
    );
    let mut p = Parser {
        s: expression.as_bytes(),
        pos: 0,
        nodes: 0,
    };
    let n = p.expr(0, 0)?;
    p.ws();
    ensure!(p.pos == p.s.len(), "unsupported arithmetic token");
    Ok(n)
}
impl Parser<'_> {
    fn ws(&mut self) {
        while self.s.get(self.pos).is_some_and(u8::is_ascii_whitespace) {
            self.pos += 1;
        }
    }
    fn expr(&mut self, min_bp: u8, depth: usize) -> Result<f64> {
        self.nodes += 1;
        ensure!(
            depth <= 64 && self.nodes <= 1024,
            "arithmetic complexity cap"
        );
        self.ws();
        let c = *self
            .s
            .get(self.pos)
            .ok_or_else(|| anyhow::anyhow!("missing operand"))?;
        let mut left = if matches!(c, b'+' | b'-') {
            self.pos += 1;
            let x = self.expr(25, depth + 1)?;
            if c == b'-' { -x } else { x }
        } else if c == b'(' {
            self.pos += 1;
            let x = self.expr(0, depth + 1)?;
            self.ws();
            ensure!(
                self.s.get(self.pos) == Some(&b')'),
                "missing closing parenthesis"
            );
            self.pos += 1;
            x
        } else {
            let start = self.pos;
            while self
                .s
                .get(self.pos)
                .is_some_and(|b| b.is_ascii_digit() || *b == b'.')
            {
                self.pos += 1;
            }
            if matches!(self.s.get(self.pos), Some(b'e' | b'E')) {
                self.pos += 1;
                if matches!(self.s.get(self.pos), Some(b'+' | b'-')) {
                    self.pos += 1;
                }
                while self.s.get(self.pos).is_some_and(u8::is_ascii_digit) {
                    self.pos += 1;
                }
            }
            ensure!(self.pos > start, "numeric literal required");
            std::str::from_utf8(&self.s[start..self.pos])?.parse::<f64>()?
        };
        ensure!(left.is_finite(), "non-finite numeric value");
        loop {
            self.ws();
            let rest = &self.s[self.pos..];
            let (op, len, l, r) = if rest.starts_with(b"**") {
                ("**", 2, 30, 30)
            } else if rest.starts_with(b"//") {
                ("//", 2, 20, 21)
            } else {
                match rest.first() {
                    Some(b'+') => ("+", 1, 10, 11),
                    Some(b'-') => ("-", 1, 10, 11),
                    Some(b'*') => ("*", 1, 20, 21),
                    Some(b'/') => ("/", 1, 20, 21),
                    Some(b'%') => ("%", 1, 20, 21),
                    _ => break,
                }
            };
            if l < min_bp {
                break;
            }
            self.pos += len;
            let right = self.expr(r, depth + 1)?;
            left = match op {
                "+" => left + right,
                "-" => left - right,
                "*" => left * right,
                "/" | "//" | "%" => {
                    ensure!(right != 0.0, "division by zero");
                    match op {
                        "/" => left / right,
                        "//" => (left / right).floor(),
                        _ => left - (left / right).floor() * right,
                    }
                }
                "**" => {
                    ensure!(right.abs() <= 1000.0, "exponent cap");
                    left.powf(right)
                }
                _ => bail!("unsupported operator"),
            };
            ensure!(left.is_finite(), "arithmetic overflow/domain error");
        }
        Ok(left)
    }
}
