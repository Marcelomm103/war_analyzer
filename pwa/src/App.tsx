function App() {
  return (
    <main className="page">
      <header className="page-header">
        <span className="brand-mark" aria-hidden="true">
          W
        </span>
        <div>
          <p className="eyebrow">WAR · ASSISTENTE DE PARTIDA</p>
          <h1>Prepare a mesa.</h1>
          <p className="intro">
            Estrutura inicial da interface para acompanhar uma partida.
          </p>
        </div>
      </header>

      <section className="status-card" aria-label="Status da interface">
        <span className="status-dot" aria-hidden="true" />
        <p>
          A base visual está pronta. Sessões, voz e confirmações serão
          conectadas nas fases seguintes.
        </p>
      </section>
    </main>
  );
}

export default App;
